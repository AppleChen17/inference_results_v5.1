import multiprocessing as mp
from multiprocessing import connection as conn
import os
import logging
import time
import numpy as np
import array
import mlperf_loadgen as lg

from harness_llm.loadgen.sut import SUT, SUTConfig
from harness_llm.backends.common.utils import check_parallelism_configuration
from harness_llm.common.sorting import SortingStrategy, validate_sorting_strategy
from harness_llm.common.rpd_trace_utils import rpd_trace_range_non_timed
from harness_llm.backends.common.debug import DebugToolkit
from threading import Thread, Event
import harness_llm.backends.common.constants as constants
from harness_llm.common.config_parser import HarnessCfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__file__)

class LLMProc:
    def __init__(
        self,
        device_ids: tuple[int, ...],
        qdata_in: conn.Connection,
        qdata_out: conn.Connection,
        llm_config: dict,
        sampling_params: dict,
        engine,
        engine_version,
        benchmark,
    ):
        self.llm_config = llm_config
        self.sampling_params = sampling_params
        self.qdata_in = qdata_in
        self.qdata_out = qdata_out
        self.qstatus_out = mp.Queue()
        self.device_ids = device_ids

        log.info(f"llm_config={self.llm_config}")
        os.environ["HIP_VISIBLE_DEVICES"] = ",".join((str(i) for i in self.device_ids))

        self.llm_proc = mp.Process(
            target=engine,
            args=(
                self.device_ids,
                self.qdata_in,
                self.qdata_out,
                self.qstatus_out,
                self.llm_config,
                self.sampling_params,
                engine_version,
                benchmark
            ),
        )

        self.llm_proc.start()

    def check_llm_loaded(self) -> bool:
        while True:
            status = self.qstatus_out.get()
            if status == constants.HarnessStates.LLM_MODEL_LOAD_DONE:
                log.info(f"LLM is loaded")
                return True


class OfflineBaseSUT(SUT):
    def __init__(
            self,
            config: HarnessCfg,
            llm_config: dict,
            sampling_config: dict,
            engine
    ):
        log.info(f"Init Offline SUT")

        super().__init__(
            SUTConfig(
                dataset_path=config["harness_config"]["dataset_path"],
                total_sample_count=(
                    config["harness_config"]["total_sample_count"]
                    if "total_sample_count" in config["harness_config"]
                    else 24576
                ),
                model_max_length=(
                    config["harness_config"]["model_max_length"]
                    if "model_max_length" in config["harness_config"]
                    else None
                ),
                debug=False,
            )
        )

        self.engine = engine
        self.engine_version = config.get_with_default("engine_version", "sync")
        self.benchmark = config["benchmark_name"]
        self.llm_config: dict = llm_config
        self.harness_config: dict = config["harness_config"]
        self.sampling_params: dict = sampling_config
        self.debug_toolkit: DebugToolkit = DebugToolkit(
            harness_config = self.harness_config,
            llm_config = self.llm_config
        )

        self.tp = self.harness_config["tensor_parallelism"]
        self.pp = self.harness_config["pipeline_parallelism"]
        # TODO: get the default value based on the number of devices available & tp config.
        self.dp = self.harness_config.get("device_count", 8)
        self.effective_dp = self.dp // (self.tp * self.pp) 

        self.warmup_sample = constants.WarmUp.ENCODED_SAMPLES.get(config["benchmark_name"], None)
        self.enable_warmup = self.harness_config["enable_warmup"] and (self.warmup_sample is not None)
        self.sorting_params = self.harness_config.get("sorting", {})

        self.qdata_in_senders = []
        self.qdata_out_receivers = []
        self.qstatus_out = mp.Queue()

        self.llm_procs = []
        self.llm_objs = []
        self.warm_up_done = []

        self.sample_ids = []
        self.completion_threads = []
        self.start_t = time.time()
        self.infer_start_t = time.time()

    @rpd_trace_range_non_timed("SUT:Main")
    def init_llms(self):
        check_parallelism_configuration(self.dp, self.tp, self.pp)
        validate_sorting_strategy(self.sorting_params)
        self.sorting = SortingStrategy(self.data_object,
                                       self.harness_config["max_num_batched_tokens"])

        qdata_out = mp.Queue()
        self.qdata_out_receivers.append(qdata_out)
    
        for i in range(0, self.effective_dp):
            engine_device_size = self.tp * self.pp
            device_ids = tuple(range(engine_device_size * i, engine_device_size * (i + 1)))
            qdata_in_receiver, qdata_in_sender = mp.Pipe(False)
            self.qdata_in_senders.append(qdata_in_sender)

            self.warm_up_done.append(Event())
            llm_obj = LLMProc(
                device_ids,
                qdata_in_receiver,
                qdata_out,
                self.llm_config,
                self.sampling_params,
                self.engine,
                self.engine_version,
                self.benchmark
            )

            self.llm_objs.append(llm_obj)

        for obj in self.llm_objs:
            obj.check_llm_loaded()

    def start_completion_loop(self):
        self.completion_threads.append(Thread(target=self.completion_queue, args=(self.effective_dp,), daemon=True))
        self.completion_threads[-1].start()

    @rpd_trace_range_non_timed("SUT:Main")
    def warm_up(self):
        log.info("Running warmup")
        for i in range(self.effective_dp):
            prompt_token_ids = [self.warmup_sample] * 10
            self.qdata_in_senders[i].send((0, None, prompt_token_ids, None))
        for i in range(self.effective_dp):
            log.info(f"Waiting for server[{i}] warmup to complete...")
            self.warm_up_done[i].wait()
        log.info("Running warmup finished")

    @rpd_trace_range_non_timed("SUT:Main")
    def stop(self):
        for t in self.completion_threads:
            t.join()
        log.info(f"Total time spent with run: {time.time() - self.start_t}")

    @rpd_trace_range_non_timed("SUT:Main")
    def start(self):
        log.info(f"SUT start")
        self.init_llms()
        self.start_completion_loop()
        if self.enable_warmup:
            self.warm_up()
        self.infer_start_t = time.time()
        log.info(
            f"Time spent from start to inference start: {self.infer_start_t - self.start_t}"
        )

    @rpd_trace_range_non_timed("SUT:Main")
    def post_proc(self, response):
        start, end, output_token_ids = response
        log.info(
            f"Got item  |  start, end = {start}, {end}  |  n outputs = {len(output_token_ids)}"
        )

        if self.harness_config['debug_dump_model_output']:
            self.debug_toolkit.dump(output_token_ids)

        output_sample_ids = self.sample_ids[start:end]
        assert len(output_sample_ids) == len(output_token_ids)

        log.info(f"Signaling LoadGen output")

        try:
            for i in range(len(output_token_ids)):
                response_array = array.array(
                    "B", np.array(output_token_ids[i], np.int32).tobytes()
                )
                bi = response_array.buffer_info()
                response = [
                    lg.QuerySampleResponse(
                        output_sample_ids[i], bi[0], bi[1], len(output_token_ids[i])
                    )
                ]
                lg.QuerySamplesComplete(response)
        except:
            log.info(f"Error sending completed response to LoadGen")

    @rpd_trace_range_non_timed("SUT:Main")
    def completion_queue(self, devices):
        warm_up_in_progress_count = devices
        while True:
            try:
                response = self.qdata_out_receivers[-1].get()
                if response == constants.HarnessStates.LLM_GENERATION_DONE:
                    log.info(f"Query chunk done. Remaining GPUs: {devices}")
                    devices -= 1
                    if devices <= 0:
                        break
                else:
                    if self.enable_warmup and warm_up_in_progress_count > 0:
                        warm_up_in_progress_count -= 1
                        self.warm_up_done[warm_up_in_progress_count].set()
                    else:
                        self.post_proc(response)
            except:
                logging.exception("Exception during completion")
                break

    @rpd_trace_range_non_timed("SUT:Main")
    def issue_queries(self, query_samples):
        log.info(f"Issue queries  |  number of queries = {len(query_samples)}")
        ranges, query_samples = self.sorting.sort_samples(query_samples, self.sorting_params, self.effective_dp)
        self.sample_ids = [query_samples[i].id for i in range(len(query_samples))]
        prompt_token_ids = [
            self.data_object.input_ids[query_samples[i].index]
            for i in range(len(query_samples))
        ]

        log.info(
            f"Converted queries to prompt tokens  |  number of queries = {len(prompt_token_ids)}"
        )

        stop_token_ids = None
        if self.data_object.stop_ids:
            stop_token_ids = [
                self.data_object.stop_ids[query_samples[i].index]
                for i in range(len(query_samples))
            ]           

        for i, (start, end) in enumerate(ranges):
            self.qdata_in_senders[i].send((start, end, prompt_token_ids[start:end], stop_token_ids[start:end] if stop_token_ids else None))
            log.info(f"Put prompt tokens in pipe #{i}")

        for i in range(self.effective_dp):
            self.qdata_in_senders[i].send(None)
            self.qdata_in_senders[i].close()
