from harness_llm.backends.common.server_sut import ServerBaseSUT
from harness_llm.backends.vllm.sync_server import SyncServer
from harness_llm.backends.vllm.async_server import AsyncServer

class SyncServerVLLMSUT(ServerBaseSUT):

    def __init__(self, config: dict, llm_config: dict, sampling_config: dict):
        super().__init__(
            config=config,
            llm_config=llm_config,
            sampling_config=sampling_config, 
            engine=SyncServer
        )

class AsyncServerVLLMSUT(ServerBaseSUT):

    def __init__(self, config: dict, llm_config: dict, sampling_config: dict):
        super().__init__(
            config=config,
            llm_config=llm_config,
            sampling_config=sampling_config, 
            engine=AsyncServer
        )
