from ray.util.queue import Queue
from ray.util.placement_group import placement_group
import os
import gc
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import ray
from harness_llm.backends.vllm.ray.distributed_sync_server import DistributedSyncServer
from harness_llm.loadgen.sut import SUT, SUTConfig
import harness_llm.common.logging as logging
import multiprocessing as mp

import harness_llm.backends.common.constants as constants
from threading import Thread, Event
import mlperf_loadgen as lg
import array
import numpy as np
import time
from ray.runtime_env import RuntimeEnv
import threading

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__file__)


class DistributedServerSUT(SUT):

    def __init__(self, config: dict, engine):
        super().__init__(
            SUTConfig(
                model = config["llm_config"]["model"],
                dataset_path = config["harness_config"]["dataset_path"],
                total_sample_count = config["harness_config"]["total_sample_count"],
                model_max_length = None,
                debug=False
            )
        )

        self.env_config = {k: str(v) for k, v in config.env_config.items()}  

        self.tensor_parallelism = config["harness_config"]["tensor_parallelism"]
        self.pipeline_parallelism = config["harness_config"]["pipeline_parallelism"]
        self.device_count = config["harness_config"]["device_count"]
        self.effective_dp = self.device_count // (self.tensor_parallelism * self.pipeline_parallelism)

        self.engine_config: dict = config["llm_config"]
        self.harness_config: dict = config["harness_config"]
        self.sampling_config: dict = config["sampling_params"]

        self.qdata_in = []
        self.qdata_out = []
        self.qstatus_out = Queue()

        self.engine = engine
        self.engine_actors = []
        self.output_collector_threads = []
        self.device_counter = 0

        self.servers = {}


        self.warmup_sample = constants.WarmUp.ENCODED_SAMPLES.get(config["benchmark_name"], None)
        self.enable_warmup = self.harness_config["enable_warmup"] and (self.warmup_sample is not None)
        self.warm_up_done = []
        
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

        # The GC is going to be called after certain number of samples
        self.HARNESS_GC_LIMIT = int(os.getenv('HARNESS_GC_LIMIT', 0))
        self.sample_count = 0
        self.is_gc_limit_specified = self.HARNESS_GC_LIMIT > 0
        if self.is_gc_limit_specified:
            gc.collect()
            gc.disable()


    def start(self):
        ray.init(address="auto", ignore_reinit_error=True)

        for i in range(0, self.effective_dp):
            pg = placement_group([{"GPU": 1, "CPU" : 1}] * self.tensor_parallelism, strategy="STRICT_PACK")

            qdata_in = Queue()
            qdata_out = Queue()
            qstatus_out = Queue()

            ray.get(pg.ready())
            server = self.engine.options(
                                    runtime_env={"env_vars": self.env_config},
                                    num_gpus=self.tensor_parallelism,
                                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                                        placement_group=pg,
                                        placement_group_capture_child_tasks=True
                                        )
                                ).remote(
                                    devices = i,
                                    qdata_in = qdata_in, 
                                    qdata_out = qdata_out,
                                    qstatus_out = qstatus_out,
                                    llm_config = self.engine_config, 
                                    sampling_params = self.sampling_config
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

            self.servers[i]["server"].start.remote()
            self.warm_up_done.append(threading.Event())
            self.output_collector_threads.append(threading.Thread(
                target=self.send_outputs, args=([qdata_out, i]), daemon=True
            ))
            self.output_collector_threads[-1].start()

        for index in self.servers:
            while True:
                log.info(f"i={index} | Polling server...")
                if self.servers[index]["server"].is_running.remote():
                    log.info(f"i={index} | Server is ready")
                    break
                else:
                    time.sleep(10)

        if self.enable_warmup:
            self.warm_up()


    def warm_up(self):
        log.info(f"Running warmup")
        for i in range(self.effective_dp):
            items = [("0", self.warmup_sample, "WARMUP_QUERY_TYPE")]
            self.servers[i]["qdata_in"].put(items)
        for i in range(self.effective_dp):
            log.info(f"Waiting for server[{i}] warmup to complete...")
            self.warm_up_done[i].wait()
        log.info("Running warmup finished")


    def send_samples(self, samples):
        items = [
            (str(sample.id), self.data_object.input_ids[sample.index], self.data_object.query_types[sample.index])
            for sample in samples
        ]
        device_id = self.get_next_device()
        self.servers[device_id]["qdata_in"].put(items)
        self.servers[device_id]["sent"] += 1    


    def issue_queries(self, query_samples):
        num_samples = len(query_samples)
        # log.info(f"[Server] Received {num_samples} samples")
        self.sample_count += num_samples
        if self.is_gc_limit_specified and self.sample_count >= self.HARNESS_GC_LIMIT:
            gc.collect()
            self.sample_count = 0
        for sample in query_samples:
            self.send_sample(sample)

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

    def warmup_finished(self, token_ids, device_id):
        if token_ids is None:
            self.warm_up_done[device_id].set()
            return True
        return False

    def send_outputs(self, qdata_out, device_id):
        self.log("Collecting outputs started...")
        is_warmup_finished = False
        while True:
            response = qdata_out.get()
            if response is None:
                break
            if self.enable_warmup and not is_warmup_finished:
                is_warmup_finished = self.warmup_finished(response[1], device_id)
            else:
                self.post_proc(response, device_id)
            # if not self.stopped:
            #    self.print_finished()

    def stop(self):
        for index in self.servers:
            self.servers[index]["qdata_in"].put(None)
        self.stopped = True
        time.sleep(10)

    def next_device_id(self):
        next_div_id = self.device_counter
        self.device_counter = (self.device_counter + 1) % len(self.servers)
        return next_div_id

    def next_best_device_id(self):
        next_div_id = 0
        min_queue = 1_000_000_000
        for d in range(self.effective_dp):
            diff = self.servers[d]["sent"] - self.servers[d]["finished"]
            if diff < min_queue:
                min_queue = diff
                next_div_id = d
        return next_div_id

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

    def send_sample(self, sample):
        prompt_token_ids = self.data_object.input_ids[sample.index]
        query_types = self.data_object.query_types[sample.index]
        device_id = self.get_next_device()
        if self.harness_config["schedule_algo"] == "shortest_queue_with_tokens":
            window_size = self.harness_config["load_balance_window_size"]
            if len(self.servers[device_id]["tokens_in"]) > window_size:
                # 10 is used as the window_size for this algorithm. This can be tuned potentially for better perf
                self.servers[device_id]["tokens_in"].pop(0)
            self.servers[device_id]["tokens_in"].append(len(prompt_token_ids))
        self.servers[device_id]["qdata_in"].put(
            [(str(sample.id), prompt_token_ids, query_types)]
        )
        self.servers[device_id]["sent"] += 1

    def log(self, message: str):
        log.info(f"SUT - {message}")


class Sample:
    def __init__(self, index):
        self.index = index
        self.id = index

class DistributedSyncServerSUT(DistributedServerSUT):
    def __init__(self, config: dict):
        super().__init__(
            config=config,            
            engine=DistributedSyncServer
        )