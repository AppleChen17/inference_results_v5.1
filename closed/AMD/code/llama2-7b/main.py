import multiprocessing as mp
import os
import mlperf_loadgen as lg
from harness_llm.backends.vllm.sut_offline import OfflineVLLMSUT
from harness_llm.backends.vllm.server_sut import (
    SyncServerVLLMSUT,
    AsyncServerVLLMSUT
)
from harness_llm.loadgen.sut import SUT
from harness_llm.common.config_parser import HarnessCfg
from resource_checker import ResourceChecker
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__file__)

scenario_map = {
    "offline": lg.TestScenario.Offline,
    "server": lg.TestScenario.Server,
}

sample_count_map = {
    "llama2-70b": 24576,
    "llama2-70b-interactive": 24576,
    "mixtral-8x7b": 15000,
    "llama3_1-405b": 8313,
}

def get_mlperf_test_settings(
    benchmark: str, scenario: str, test_mode: str, harness_config
) -> lg.TestSettings:
    "Returns the test settings needed for mlperf"
    settings = lg.TestSettings()
    settings.scenario = scenario_map[scenario.lower()]
    # settings.FromConfig(harness_config.mlperf_conf_path, benchmark.lower(), scenario)
    settings.FromConfig(harness_config.user_conf_path, benchmark.lower(), scenario.capitalize())

    if harness_config.target_qps > 0:
        settings.offline_expected_qps = harness_config.target_qps
        settings.server_target_qps = harness_config.target_qps
        log.warning(
            f"Overriding default QPS with {harness_config.target_qps}"
        )

    if harness_config.total_sample_count != sample_count_map[benchmark.lower()]:
        settings.min_query_count = harness_config.total_sample_count
        settings.max_query_count = harness_config.total_sample_count
        log.warning(
            f"Overriding default sample count with {harness_config.total_sample_count}"
        )

    if harness_config.duration_sec != -1:
        time_ms = harness_config.duration_sec * 1000
        settings.min_duration_ms = time_ms
        settings.max_duration_ms = time_ms
        log.warning(
            f"Overriding default duration with {time_ms} ms"
        )

    if test_mode.lower() == "accuracy":
        settings.mode = lg.TestMode.AccuracyOnly
        log.warning(
            "Accuracy run will generate the accuracy logs, but the evaluation of the log is not completed yet"
        )
    else:
        settings.mode = lg.TestMode.PerformanceOnly

    return settings


def get_mlperf_log_settings(harness_config) -> lg.LogOutputSettings:
    # create the log dir if not exist.
    os.makedirs(harness_config.output_log_dir, exist_ok=True)
    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = harness_config.output_log_dir
    log_output_settings.copy_summary_to_stdout = True
    log_settings = lg.LogSettings()
    log_settings.log_output = log_output_settings
    log_settings.enable_trace = harness_config.enable_log_trace
    return log_settings


def get_sut(scenario: str, engine_version: str, backend: str, conf: dict) -> SUT:
    """
    This returns an instance of SUT depends on the inputs.
    """
    llm_config = conf["llm_config"]
    sampling_config = conf["sampling_params"]
    
    if scenario.lower() == "offline":
        if backend == "sglang":
            from harness_llm.backends.sglang.offline_sut import OfflineSGLangSUT

            conf["harness_config"]["tensor_parallelism"] = llm_config["tp_size"]
            conf["harness_config"]["pipeline_parallelism"] = 1
            conf["harness_config"]["max_num_batched_tokens"] = 0
            return OfflineSGLangSUT(
                config=conf,
                llm_config=llm_config,
                sampling_config=sampling_config
            )

        if backend == "ray":
            from harness_llm.backends.vllm.ray.distributed_offline_sut import DistributedOfflineSUT

            conf["harness_config"]["tensor_parallelism"] = llm_config["tensor_parallel_size"]
            conf["harness_config"]["pipeline_parallelism"] = llm_config["pipeline_parallel_size"]
            conf["harness_config"]["max_num_batched_tokens"] = llm_config["max_num_batched_tokens"]
            return DistributedOfflineSUT(
                config=conf
            )

        conf["harness_config"]["tensor_parallelism"] = llm_config["tensor_parallel_size"]
        conf["harness_config"]["pipeline_parallelism"] = llm_config["pipeline_parallel_size"]
        conf["harness_config"]["max_num_batched_tokens"] = llm_config["max_num_batched_tokens"]
        return OfflineVLLMSUT(
            config=conf,
            llm_config=llm_config,
            sampling_config=sampling_config
        )
    elif scenario.lower() == "server":
        if backend == "sglang":
            from harness_llm.backends.sglang.server_sut import AsyncServerSGLangSUT

            conf["harness_config"]["tensor_parallelism"] = llm_config["tp_size"]
            conf["harness_config"]["pipeline_parallelism"] = 1
            return AsyncServerSGLangSUT(
                conf,
                llm_config,
                sampling_config
            )

        if backend == "ray":
            from harness_llm.backends.vllm.ray.distributed_server_sut import DistributedSyncServerSUT

            conf["harness_config"]["tensor_parallelism"] = llm_config["tensor_parallel_size"]
            conf["harness_config"]["pipeline_parallelism"] = llm_config["pipeline_parallel_size"]
            conf["harness_config"]["max_num_batched_tokens"] = llm_config["max_num_batched_tokens"]
            return DistributedSyncServerSUT(
                config=conf
            )    

        if engine_version.lower() == "sync":
            conf["harness_config"]["tensor_parallelism"] = llm_config["tensor_parallel_size"]
            conf["harness_config"]["pipeline_parallelism"] = llm_config["pipeline_parallel_size"]
            return SyncServerVLLMSUT(
                conf,
                llm_config,
                sampling_config
            )
        elif engine_version.lower() == "async":
            conf["harness_config"]["tensor_parallelism"] = llm_config["tensor_parallel_size"]
            conf["harness_config"]["pipeline_parallelism"] = llm_config["pipeline_parallel_size"]
            return AsyncServerVLLMSUT(
                conf,
                llm_config, 
                sampling_config
            )
        else:
            raise ValueError(f"Unsupported engine version is passed - {engine_version}")
    else:
        raise ValueError(f"Unsupported scenario is passed - {scenario}")


def set_mlperf_envs(env_config: dict):
    print(f"{env_config=}", flush=True)
    for env, val in env_config.items():
        if val is not None:
            os.environ[env] = str(val)
            log.info(f"Setting {env} to {val}")


def run_mlperf_tests(conf) -> None:
    """
    A main entry point to run the mlperf tests.
    """

    # Set mlperf test and log settings
    test_settings = get_mlperf_test_settings(
        benchmark=conf.benchmark_name,
        scenario=conf.scenario,
        test_mode=conf.test_mode,
        harness_config=conf.harness_config,
    )

    log_settings = get_mlperf_log_settings(harness_config=conf.harness_config)

    ResourceChecker().check(dict(conf.config))

    # Set ENVs for mlperf
    set_mlperf_envs(conf["env_config"])

    # Instantiate SUT
    sut = get_sut(
        scenario=conf.scenario, 
        engine_version=conf.engine_version,
        backend=conf.backend, 
        conf=conf
    )

    log.info("Instantiating SUT")
    sut.start()
    lgSUT = lg.ConstructSUT(sut.issue_queries, sut.flush_queries)
    print(conf.to_yaml())
    lg.StartTestWithLogSettings(lgSUT, sut.qsl, test_settings, log_settings)
    log.info("Completed benchmark run")
    sut.stop()
    log.info("Run Completed!")
    log.info("Destroying SUT...")
    lg.DestroySUT(lgSUT)
    log.info("Destroying QSL...")
    lg.DestroyQSL(sut.qsl)


def run_from_cli() -> None:
    harnessCfg = HarnessCfg().create_from_cli()
    run_mlperf_tests(harnessCfg)


def run_from_optuna(config_path, config_name, backend, overrides) -> None:
    harnessCfg = HarnessCfg().create_from_optuna(config_path, config_name, backend, overrides)
    run_mlperf_tests(harnessCfg)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    log.info(f"mp.get_context:{mp.get_context()}")
    run_from_cli()
