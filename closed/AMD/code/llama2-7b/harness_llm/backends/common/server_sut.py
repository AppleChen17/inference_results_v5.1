import logging
import mlperf_loadgen as lg
import multiprocessing as mp
import time
import numpy as np
import array

from harness_llm.loadgen.sut import SUT
from harness_llm.common.rpd_trace_utils import rpd_trace_range, rpd_trace_range_non_timed
import threading
import sys
from datetime import datetime
import gc
import os
from harness_llm.backends.common.debug import DebugToolkit
from harness_llm.backends.common.utils import check_parallelism_configuration
from harness_llm.loadgen.sut import SUT, SUTConfig
import harness_llm.backends.common.constants as constants

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__file__)


class ServerBaseSUT(SUT):
    def __init__(
            self, config: dict,
            llm_config: dict,
            sampling_config: dict,
            engine
    ):
        log.info(f"Init SUTvLLMServer")

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


        self.engine_class = engine
        self.benchmark = config["benchmark_name"]
        self.llm_config: dict = llm_config
        self.harness_config: dict = config["harness_config"]
        self.sampling_params: dict = sampling_config
        self.debug_toolkit: DebugToolkit = DebugToolkit(
            harness_config = self.harness_config,
            llm_config = self.llm_config
        )

        self.tp = config["harness_config"]["tensor_parallelism"]
        self.pp = config["harness_config"]["pipeline_parallelism"]
        # TODO: get the default value based on the number of devices available & tp config.
        self.dp = self.harness_config.get("device_count", 8)

        self.effective_dp = self.dp

        self.servers = {}
        self.output_collector_threads = []
        self.device_counter = 0

        self.total_sample_count = self.harness_config["total_sample_count"]

        assert self.harness_config["schedule_algo"] in ["shortest_queue_with_tokens", "shortest_queue", "round_robin"], f'Unsupported schedule algo: {self.harness_config["schedule_algo"]}'

        if self.harness_config["schedule_algo"] == "shortest_queue_with_tokens":
            self.get_next_device =  self.next_best_device_id_with_tokens
        elif  self.harness_config["schedule_algo"] == "shortest_queue":
            self.get_next_device = self.next_best_device_id
        else:
            self.get_next_device = self.next_device_id


        self.n_finished = 0
        self.n_finished_first = 0
        self.stopped = False
        self.response_buffer = {}
        check_parallelism_configuration(self.dp, self.tp, self.pp)
        self.effective_dp = self.dp // (self.tp * self.pp)
        # The GC is going to be called after certain number of samples
        self.HARNESS_GC_LIMIT = int(os.getenv('HARNESS_GC_LIMIT', 0))
        self.sample_count = 0
        self.is_gc_limit_specified = self.HARNESS_GC_LIMIT > 0
        if self.is_gc_limit_specified:
            gc.collect()
            gc.disable()


    @rpd_trace_range_non_timed("SUT:Main")
    def start(self):

        for i in range(0, self.effective_dp):
            engine_device_size = self.tp * self.pp
            devices = range(engine_device_size * i, engine_device_size * (i + 1))

            qdata_in = mp.Queue()
            qdata_out = mp.Queue()
            qstatus_out = mp.Queue()

            server = self.engine_class(
                devices,
                qdata_in,
                qdata_out,
                qstatus_out,
                self.llm_config,
                self.sampling_params,
                self.benchmark,
                self.harness_config["enable_warmup"]
            )

            self.servers[i] = {
                "server": server,
                "qdata_in": qdata_in,
                "qdata_out": qdata_out,
                "qstatus_out": qstatus_out,
                "sent": 0,
                "finished": 0,
                "tokens_in":[],
            }

            self.servers[i]["server"].start()
            self.output_collector_threads.append(threading.Thread(
                target=self.send_outputs, args=([qdata_out, i]), daemon=True
            ))
            self.output_collector_threads[-1].start()

        for index in self.servers:
            while True:
                log.info(f"i={index} | Polling server...")
                if self.servers[index]["server"].is_running():
                    log.info(f"i={index} | Server is ready")
                    break
                else:
                    time.sleep(10)

    @rpd_trace_range("SUT:Main")
    def send_samples(self, samples):
        items = [
            (str(sample.id), self.data_object.input_ids[sample.index], self.data_object.stop_ids[sample.index] if self.data_object.stop_ids else None)
            for sample in samples
        ]
        device_id = self.get_next_device()
        self.servers[device_id]["qdata_in"].put_nowait(items)
        self.servers[device_id]["sent"] += len(items)

    @rpd_trace_range("SUT:Main")
    def issue_queries(self, query_samples):
        num_samples = len(query_samples)
        # log.info(f"[Server] Received {num_samples} samples")
        self.sample_count += num_samples
        if self.is_gc_limit_specified and self.sample_count >= self.HARNESS_GC_LIMIT:
            gc.collect()
            self.sample_count = 0
        for sample in query_samples:
            self.send_sample(sample)

        if self.harness_config['debug_record_sample_latencies']:
            for sample in query_samples:
                self.debug_toolkit.record_sample_latencies(sample.id, None, input_token_count=len(self.data_object.input_ids[sample.index]))

    def print_finished(self):
        # time
        now = datetime.now()
        now_mon = "0"
        if now.month < 10:
            now_mon += str(now.month)
        else:
            now_mon = str(now.month)

        now_day = "0"
        if now.day < 10:
            now_day += str(now.day)
        else:
            now_day = str(now.day)

        now_hr = "0"
        if now.hour < 10:
            now_hr += str(now.hour)
        else:
            now_hr = str(now.hour)

        now_min = "0"
        if now.minute < 10:
            now_min += str(now.minute)
        else:
            now_min = str(now.minute)

        now_sec = "0"
        if now.second < 10:
            now_sec += str(int(now.second))
        else:
            now_sec = str(int(now.second))

        tm = (
            str(now.year)
            + "-"
            + now_mon
            + "-"
            + now_day
            + " "
            + now_hr
            + ":"
            + now_min
            + ":"
            + now_sec
            + " INFO     SUT - "
        )
        msg = (
            "\r"
            + tm
            + "Processed prompts: "
            + str(self.n_finished)
            + " first tokens: "
            + str(self.n_finished_first)
            + " "
            + " | ".join((str(d)+":"+str(self.servers[d]["sent"])+"/"+str(self.servers[d]["finished"])+" q:"+str(self.servers[d]["sent"]-self.servers[d]["finished"]) for d in range(self.effective_dp)))
            + " "
        )
        sys.stdout.write(msg)
        sys.stdout.flush()

    @rpd_trace_range("SUT:Main")
    def post_proc(self, response, device_id):
        sample_id = int(response[0])
        token_ids = response[1]
        finished = token_ids is None
        if finished:
            if self.harness_config['debug_dump_model_output']:
                self.debug_toolkit.dump([self.response_buffer[sample_id]])

            response_array = array.array(
                "B", np.array(self.response_buffer[sample_id], np.int32).tobytes()
            )
            bi = response_array.buffer_info()
            response = [
                lg.QuerySampleResponse(
                    sample_id, bi[0], bi[1], len(self.response_buffer[sample_id])
                )
            ]
            lg.QuerySamplesComplete(response)
            del self.response_buffer[sample_id]
            self.n_finished += 1
            self.servers[device_id]["finished"] += 1
        elif sample_id not in self.response_buffer:
            self.response_buffer[sample_id] = list(token_ids)
            response_array = array.array("B", np.array(token_ids, np.int32).tobytes())
            bi = response_array.buffer_info()
            response = [lg.QuerySampleResponse(sample_id, bi[0], bi[1], len(token_ids))]
            lg.FirstTokenComplete(response)
            self.n_finished_first += 1
        else:
            self.response_buffer[sample_id].extend(token_ids)

        if self.harness_config['debug_record_sample_latencies']:
            self.debug_toolkit.record_sample_latencies(sample_id, token_ids)

    def send_outputs(self, qdata_out, device_id):
        self.log("Collecting outputs started...")
        while True:
            response = qdata_out.get()
            if response is None:
                break
            self.post_proc(response, device_id)
            # if not self.stopped:
            #    self.print_finished()

    @rpd_trace_range_non_timed("SUT:Main")
    def stop(self):
        for index in self.servers:
            self.servers[index]["qdata_in"].put(None)
        self.stopped = True
        time.sleep(10)

    @rpd_trace_range("SUT:Main")
    def next_device_id(self):
        next_div_id = self.device_counter
        self.device_counter = (self.device_counter + 1) % len(self.servers)
        return next_div_id

    @rpd_trace_range("SUT:Main")
    def next_best_device_id(self):
        next_div_id = 0
        min_queue = 1_000_000_000
        for d in range(self.effective_dp):
            diff = self.servers[d]["sent"] - self.servers[d]["finished"]
            if diff < min_queue:
                min_queue = diff
                next_div_id = d
        return next_div_id

    @rpd_trace_range("SUT:Main")
    def next_best_device_id_with_tokens(self):
        next_div_id = 0
        min_queue = 1_000_000_000
        for d in range(self.effective_dp):
            num_tokens = sum(self.servers[d]['tokens_in'])
            token_weight = self.harness_config["load_balance_token_weight"]
            diff = (self.servers[d]["sent"] - self.servers[d]["finished"]) + token_weight*num_tokens
            #This is of the form y = theta_1*x_1 + theta_2*x_2, a linear combination of the two variables.
            #theta_1 = 1 is used but could be tuned for some better perf
            #theta_2 = 0.02 is a tuned value.

            if diff < min_queue:
                min_queue = diff
                next_div_id = d
        return next_div_id

    @rpd_trace_range("SUT:Main")
    def send_sample(self, sample):
        prompt_token_ids = self.data_object.input_ids[sample.index]
        device_id = self.get_next_device()
        if self.harness_config["schedule_algo"] == "shortest_queue_with_tokens":
            window_size = self.harness_config["load_balance_window_size"]
            if len(self.servers[device_id]["tokens_in"]) > window_size:
                # 10 is used as the window_size for this algorithm. This can be tuned potentially for better perf
                self.servers[device_id]["tokens_in"].pop(0)
            self.servers[device_id]["tokens_in"].append(len(prompt_token_ids))
        self.servers[device_id]["qdata_in"].put_nowait(
            [(str(sample.id), prompt_token_ids, self.data_object.stop_ids[sample.index] if self.data_object.stop_ids else None)]
        )
        self.servers[device_id]["sent"] += 1

    def log(self, message: str):
        log.info(f"SUT - {message}")


class Sample:
    def __init__(self, index):
        self.index = index
        self.id = index
