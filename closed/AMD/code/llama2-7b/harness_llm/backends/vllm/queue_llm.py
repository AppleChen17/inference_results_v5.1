import warnings
from contextlib import contextmanager
from typing import (Any, ClassVar, Dict, List, Optional, Sequence,
                    Union, cast, overload)

from tqdm import tqdm

from vllm.engine.arg_utils import EngineArgs, TaskOption
from vllm.inputs import PromptType, TextPrompt, TokensPrompt
from vllm.inputs.parse import parse_and_batch_prompt
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.guided_decoding.guided_fields import (
    GuidedDecodingRequest)
from vllm.outputs import EmbeddingRequestOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import (GuidedDecodingParams, SamplingParams)
from vllm.transformers_utils.tokenizer import (AnyTokenizer,
                                               get_cached_tokenizer)
from vllm.transformers_utils.tokenizer_group import TokenizerGroup
from vllm.usage.usage_lib import UsageContext
from vllm.utils import Counter, deprecate_args

#if envs.VLLM_USE_V1:
#    from vllm.v1.engine.llm_engine import LLMEngine  # type: ignore
#else:
from vllm.engine.llm_engine import LLMEngine  # type: ignore
import multiprocessing as mp
import queue
import gc
import os

logger = init_logger(__name__)

HARNESS_GC_LIMIT = int(os.getenv('HARNESS_GC_LIMIT', 0))

class QueueLLM:
    """An LLM for generating texts from given prompts and sampling parameters.

    This class includes a tokenizer, a language model (possibly distributed
    across multiple GPUs), and GPU memory space allocated for intermediate
    states (aka KV cache). Given a batch of prompts and sampling parameters,
    this class generates texts from the model, using an intelligent batching
    mechanism and efficient memory management.

    Args:
        model: The name or path of a HuggingFace Transformers model.
        tokenizer: The name or path of a HuggingFace Transformers tokenizer.
        tokenizer_mode: The tokenizer mode. "auto" will use the fast tokenizer
            if available, and "slow" will always use the slow tokenizer.
        skip_tokenizer_init: If true, skip initialization of tokenizer and
            detokenizer. Expect valid prompt_token_ids and None for prompt
            from the input.
        trust_remote_code: Trust remote code (e.g., from HuggingFace) when
            downloading the model and tokenizer.
        tensor_parallel_size: The number of GPUs to use for distributed
            execution with tensor parallelism.
        dtype: The data type for the model weights and activations. Currently,
            we support `float32`, `float16`, and `bfloat16`. If `auto`, we use
            the `torch_dtype` attribute specified in the model config file.
            However, if the `torch_dtype` in the config is `float32`, we will
            use `float16` instead.
        quantization: The method used to quantize the model weights. Currently,
            we support "awq", "gptq", and "fp8" (experimental).
            If None, we first check the `quantization_config` attribute in the
            model config file. If that is None, we assume the model weights are
            not quantized and use `dtype` to determine the data type of
            the weights.
        revision: The specific model version to use. It can be a branch name,
            a tag name, or a commit id.
        tokenizer_revision: The specific tokenizer version to use. It can be a
            branch name, a tag name, or a commit id.
        seed: The seed to initialize the random number generator for sampling.
        gpu_memory_utilization: The ratio (between 0 and 1) of GPU memory to
            reserve for the model weights, activations, and KV cache. Higher
            values will increase the KV cache size and thus improve the model's
            throughput. However, if the value is too high, it may cause out-of-
            memory (OOM) errors.
        swap_space: The size (GiB) of CPU memory per GPU to use as swap space.
            This can be used for temporarily storing the states of the requests
            when their `best_of` sampling parameters are larger than 1. If all
            requests will have `best_of=1`, you can safely set this to 0.
            Otherwise, too small values may cause out-of-memory (OOM) errors.
        cpu_offload_gb: The size (GiB) of CPU memory to use for offloading
            the model weights. This virtually increases the GPU memory space
            you can use to hold the model weights, at the cost of CPU-GPU data
            transfer for every forward pass.
        enforce_eager: Whether to enforce eager execution. If True, we will
            disable CUDA graph and always execute the model in eager mode.
            If False, we will use CUDA graph and eager execution in hybrid.
        max_seq_len_to_capture: Maximum sequence len covered by CUDA graphs.
            When a sequence has context length larger than this, we fall back
            to eager mode. Additionally for encoder-decoder models, if the
            sequence length of the encoder input is larger than this, we fall
            back to the eager mode.
        disable_custom_all_reduce: See ParallelConfig
        **kwargs: Arguments for :class:`~vllm.EngineArgs`. (See
            :ref:`engine_args`)

    Note:
        This class is intended to be used for offline inference. For online
        serving, use the :class:`~vllm.AsyncLLMEngine` class instead.
    """

    DEPRECATE_LEGACY: ClassVar[bool] = False
    """A flag to toggle whether to deprecate the legacy generate/encode API."""

    DEPRECATE_INIT_POSARGS: ClassVar[bool] = True
    """
    A flag to toggle whether to deprecate positional arguments in
    :meth:`LLM.__init__`.
    """

    @classmethod
    @contextmanager
    def deprecate_legacy_api(cls):
        cls.DEPRECATE_LEGACY = True

        yield

        cls.DEPRECATE_LEGACY = False

    @deprecate_args(
        start_index=2,  # Ignore self and model
        is_deprecated=lambda: QueueLLM.DEPRECATE_INIT_POSARGS,
        additional_message=(
            "All positional arguments other than `model` will be "
            "replaced with keyword arguments in an upcoming version."),
    )
    def __init__(
        self,
        model: str,
        tokenizer: Optional[str] = None,
        tokenizer_mode: str = "auto",
        skip_tokenizer_init: bool = False,
        trust_remote_code: bool = False,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        seed: int = 0,
        gpu_memory_utilization: float = 0.9,
        swap_space: float = 4,
        cpu_offload_gb: float = 0,
        enforce_eager: Optional[bool] = None,
        max_seq_len_to_capture: int = 8192,
        disable_custom_all_reduce: bool = False,
        disable_async_output_proc: bool = False,
        mm_processor_kwargs: Optional[Dict[str, Any]] = None,
        # After positional args are removed, move this right below `model`
        task: TaskOption = "auto",
        *,
        input_queue: mp.Queue = None,
        result_queue: mp.Queue = None,
        sampling_params_config: dict = {"temparature": 0.0, "max_tokens": 1024},
        **kwargs,
    ) -> None:
        '''
        LLM constructor.

        Note: if enforce_eager is unset (enforce_eager is None)
        it defaults to False.
        '''

        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True
            
        self.warmup_in_progress = False
        self.sampling_params_config = sampling_params_config
        self.stop_seq_ids_config: dict = None
        if "stop_seq_ids_config" in self.sampling_params_config.keys():
            self.stop_seq_ids_config = self.sampling_params_config['stop_seq_ids_config']
            del self.sampling_params_config['stop_seq_ids_config']
            
        engine_args = EngineArgs(
            model=model,
            task=task,
            tokenizer=tokenizer,
            tokenizer_mode=tokenizer_mode,
            skip_tokenizer_init=skip_tokenizer_init,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            quantization=quantization,
            revision=revision,
            tokenizer_revision=tokenizer_revision,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            swap_space=swap_space,
            cpu_offload_gb=cpu_offload_gb,
            enforce_eager=enforce_eager,
            max_seq_len_to_capture=max_seq_len_to_capture,
            disable_custom_all_reduce=disable_custom_all_reduce,
            disable_async_output_proc=disable_async_output_proc,
            mm_processor_kwargs=mm_processor_kwargs,
            **kwargs,
        )
        self.llm_engine = LLMEngine.from_engine_args(
            engine_args, usage_context=UsageContext.LLM_CLASS)
        self.request_counter = Counter()

        self.input_queue = input_queue
        self.result_queue = result_queue
        
        self.finish = False

        # The GC is going to be called after certain number of steps
        self.step_count = 0
        self.is_gc_limit_specified = HARNESS_GC_LIMIT > 0
        if self.is_gc_limit_specified:
            gc.collect()
            gc.disable()

    def get_tokenizer(self) -> AnyTokenizer:
        return self.llm_engine.get_tokenizer_group(TokenizerGroup).tokenizer

    def set_tokenizer(self, tokenizer: AnyTokenizer) -> None:
        tokenizer_group = self.llm_engine.get_tokenizer_group(TokenizerGroup)

        # While CachedTokenizer is dynamic, have no choice but
        # compare class name. Misjudgment will arise from
        # user-defined tokenizer started with 'Cached'
        if tokenizer.__class__.__name__.startswith("Cached"):
            tokenizer_group.tokenizer = tokenizer
        else:
            tokenizer_group.tokenizer = get_cached_tokenizer(tokenizer)

    def start(self, use_tqdm: bool = True) -> None:
        self._pull_tokens_from_input_queue(block=True)
        self._run_engine(use_tqdm=use_tqdm)

    def _pull_all_tokens_from_input_queue(self, block: bool = True):
        while self._pull_tokens_from_input_queue(block=False):
            pass
        if block:
            self._pull_tokens_from_input_queue(block)

    def _pull_tokens_from_input_queue(self, block: bool = True):
        try:
            input = self.input_queue.get() if block else self.input_queue.get_nowait()
            if input is None:
                self.finish = True
            else:
                for sample_id, token_ids, stop_ids in input:
                    inputs = self._convert_v1_inputs(
                        prompts=None,
                        prompt_token_ids=token_ids,
                    )
                    sampling_params = SamplingParams(**self.sampling_params_config)
                    if stop_ids:
                        sampling_params.stop_seq_ids = tuple(stop_ids)
                    self._validate_and_add_requests(
                        prompts=inputs,
                        params=sampling_params,
                        request_id=sample_id,
                        lora_request=None,
                        prompt_adapter_request=None,
                    )
        except queue.Empty:
            pass
            return False
        except Exception as e:
            logger.error(f"Unexpected exception during pulling tokens: {e}")
            return False
        return True

    def start_profile(self) -> None:
        self.llm_engine.start_profile()

    def stop_profile(self) -> None:
        self.llm_engine.stop_profile()

    def start_warmup(self) -> None:
        self.warmup_in_progress = True
    
    def stop_warmup(self) -> None:
        self.warmup_in_progress = False

    def is_engine_empty(self) -> bool:
        return not self.llm_engine.has_unfinished_requests()

    # LEGACY
    def _convert_v1_inputs(
        self,
        prompts: Optional[Union[str, List[str]]],
        prompt_token_ids: Optional[Union[List[int], List[List[int]]]],
    ):
        # skip_tokenizer_init is now checked in engine

        if prompts is not None:
            prompts = [p["content"] for p in parse_and_batch_prompt(prompts)]
        if prompt_token_ids is not None:
            prompt_token_ids = [
                p["content"] for p in parse_and_batch_prompt(prompt_token_ids)
            ]

        num_requests = None
        if prompts is not None:
            num_requests = len(prompts)
        if prompt_token_ids is not None:
            if (num_requests is not None
                    and num_requests != len(prompt_token_ids)):
                raise ValueError("The lengths of prompts and prompt_token_ids "
                                 "must be the same.")

            num_requests = len(prompt_token_ids)
        if num_requests is None:
            raise ValueError("Either prompts or prompt_token_ids must be "
                             "provided.")

        parsed_prompts: List[PromptType] = []
        for i in range(num_requests):
            item: PromptType

            if prompts is not None:
                item = TextPrompt(prompt=prompts[i])
            elif prompt_token_ids is not None:
                item = TokensPrompt(prompt_token_ids=prompt_token_ids[i])
            else:
                raise AssertionError

            parsed_prompts.append(item)

        return parsed_prompts

    def _validate_and_add_requests(
        self,
        prompts: Union[PromptType, Sequence[PromptType]],
        params: Union[SamplingParams, Sequence[SamplingParams], PoolingParams,
                      Sequence[PoolingParams]],
        lora_request: Optional[Union[Sequence[LoRARequest], LoRARequest]],
        prompt_adapter_request: Optional[PromptAdapterRequest],
        request_id: str = None,
        guided_options: Optional[GuidedDecodingRequest] = None,
        priority: Optional[List[int]] = None,
    ) -> None:
        if guided_options is not None:
            warnings.warn(
                "guided_options_request is deprecated, use "
                "SamplingParams.guided_decoding instead",
                DeprecationWarning,
                stacklevel=2,
            )

        if isinstance(prompts, (str, dict)):
            # Convert a single prompt to a list.
            prompts = [prompts]

        num_requests = len(prompts)
        if isinstance(params, list) and len(params) != num_requests:
            raise ValueError("The lengths of prompts and params "
                             "must be the same.")
        if isinstance(lora_request,
                      list) and len(lora_request) != num_requests:
            raise ValueError("The lengths of prompts and lora_request "
                             "must be the same.")

        for sp in params if isinstance(params, list) else (params, ):
            if isinstance(sp, SamplingParams):
                self._add_guided_params(sp, guided_options)

                # We only care about the final output
                # sp.output_kind = RequestOutputKind.FINAL_ONLY

        # Add requests to the engine.
        for i, prompt in enumerate(prompts):
            self._add_request(
                prompt,
                params[i] if isinstance(params, Sequence) else params,
                request_id=request_id,
                lora_request=lora_request[i] if isinstance(
                    lora_request, Sequence) else lora_request,
                prompt_adapter_request=prompt_adapter_request,
                priority=priority[i] if priority else 0,
            )

    def _add_request(
        self,
        prompt: PromptType,
        params: Union[SamplingParams, PoolingParams],
        request_id: str = None,
        lora_request: Optional[LoRARequest] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
    ) -> None:
        request_id = str(next(self.request_counter)) if request_id is None else request_id
        self.llm_engine.add_request(
            request_id,
            prompt,
            params,
            lora_request=lora_request,
            prompt_adapter_request=prompt_adapter_request,
            priority=priority,
        )

    def _add_guided_params(
            self,
            params: SamplingParams,
            guided_options: Optional[GuidedDecodingRequest] = None):
        if guided_options is None:
            return params

        if params.guided_decoding is not None:
            raise ValueError("Cannot set both guided_options_request and"
                             "params.guided_decoding.")

        params.guided_decoding = GuidedDecodingParams(
            json=guided_options.guided_json,
            regex=guided_options.guided_regex,
            choice=guided_options.guided_choice,
            grammar=guided_options.guided_grammar,
            json_object=guided_options.guided_json_object,
            backend=guided_options.guided_decoding_backend,
            whitespace_pattern=guided_options.guided_whitespace_pattern)
        return params

    def _run_engine(
            self, *, use_tqdm: bool
    ) -> List[Union[RequestOutput, EmbeddingRequestOutput]]:
        # Initialize tqdm.
        if use_tqdm:
            num_requests = self.llm_engine.get_num_unfinished_requests()
            pbar = tqdm(
                total=num_requests,
                desc="Processed prompts",
                dynamic_ncols=True,
                postfix=(f"est. speed input: {0:.2f} toks/s, "
                         f"output: {0:.2f} toks/s"),
            )

        # Run the engine.
        total_in_toks = 0
        total_out_toks = 0
        request_stats = {}
        while not self.finish or self.llm_engine.has_unfinished_requests():
            block = not self.llm_engine.has_unfinished_requests() and not self.finish
            self._pull_all_tokens_from_input_queue(block=block)
            step_outputs = self.llm_engine.step()
            for output in step_outputs:
                output_len = len(output.outputs[0].token_ids)
                if output_len > 0:
                    if output.request_id not in request_stats:
                        request_stats[output.request_id] = 0
                    if not self.warmup_in_progress:
                        self.result_queue.put_nowait((output.request_id, output.outputs[0].token_ids[request_stats[output.request_id]: output_len]))
                    if output.finished:
                        # signal end of stream with None
                        if not self.warmup_in_progress:
                            self.result_queue.put_nowait((output.request_id, None))
                        del request_stats[output.request_id]
                        if use_tqdm:
                            if isinstance(output, RequestOutput):
                                # Calculate tokens only for RequestOutput
                                assert output.prompt_token_ids is not None
                                total_in_toks += len(output.prompt_token_ids)
                                in_spd = total_in_toks / pbar.format_dict["elapsed"]
                                total_out_toks += sum(
                                    len(stp.token_ids) for stp in output.outputs)
                                out_spd = (total_out_toks /
                                        pbar.format_dict["elapsed"])
                                pbar.postfix = (
                                    f"est. speed input: {in_spd:.2f} toks/s, "
                                    f"output: {out_spd:.2f} toks/s")
                            pbar.update(1)
                    else:
                        request_stats[output.request_id] = output_len

            self.step_count += 1
            if self.is_gc_limit_specified and self.step_count >= HARNESS_GC_LIMIT:
                gc.collect()
                self.step_count = 0

        if use_tqdm:
            pbar.close()

    def _is_encoder_decoder_model(self):
        return self.llm_engine.is_encoder_decoder_model()