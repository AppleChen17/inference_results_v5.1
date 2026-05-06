# MLPerf Inference 5.1 - Llama2-7B on gfx1201

The steps should be executed from the top-level AMD directory.

## Setup

### Model and Dataset

Build the docker image for model and dataset preparation by running:

```bash
bash setup/build_model_and_dataset_env.sh
```

Start the docker container for model and dataset preparation by running:

```bash
bash setup/start_model_and_dataset_env.sh
```

Inside the docker container, download the model with:

```bash
# Generate an access token on Hugging Face and set it here
HUGGINGFACE_ACCESS_TOKEN="<your HF token goes here>" python download_model.py
```

Inside the docker container, download the dataset with:

```bash
bash download_llama2_70b.sh
```

Inside the docker container, quantize the model with:

```bash
bash quantize_llama2_70b.sh
```

Exit this docker container after the model and dataset preparation is complete, because a different image is used for inference.

## Inference

### Runtime tunables

To boost the machine's performance further, execute the following script before any performance test. This only needs to be set once after a reboot.

```bash
bash setup/runtime_tunables.sh
```

### Appatainer

Set the image name for the benchmark.

```bash
export MLPERF_IMAGE_NAME=<your_rocm_mlperf_inference_image>
```

Build the image for the benchmark by running:

```bash
bash setup/build_submission_llama2_7b_apptainer.sh $MLPERF_IMAGE_NAME
```

Start the container for the benchmark by running:

```bash
bash setup/start_submission_apptainer_env.sh $MLPERF_IMAGE_NAME
```

### Running the benchmark

Run the benchmark commands inside the container.

For this task, only 1 GPU may be used for the submitted result. Therefore, please make sure the benchmark is executed with a single GPU with the setting of `harness_config.device_count = 1`.( This is also set by "DO NOT MODIFY" config in `offline_llama2_7b_gfx1201.yaml` )

Only results generated with `harness_config.device_count = 1` will be accepted for this task.

### Offline Performance

Reference command:

```bash
unset HSA_OVERRIDE_GFX_VERSION
unset ROC_ENABLE_PRE_VEGA

unset HIP_FORCE_DEV_KERNARG
unset VLLM_FP8_PADDING
unset VLLM_FP8_ACT_PADDING
unset VLLM_FP8_WEIGHT_PADDING
unset VLLM_FP8_REDUCE_CONV
unset VLLM_USE_TRITON_FLASH_ATTN
unset VLLM_SCHED_PREFILL_KVC_FREEPCT

python /lab-mlperf-inference/code/llama2-70b-99/main.py \
   --config-path /lab-mlperf-inference/code/llama2-7b/harness_llm/models/llama2-7b/ \
   --config-name offline_llama2_7b_gfx1201 \
   test_mode=performance \
   harness_config.output_log_dir=<output_log_dir>
```

### Notes

You may adjust execution-related values as needed for different test setups, including `harness_config.output_log_dir`, config and so on.

For fair comparison, keep workload-related settings such as **the model, dataset, scenario, sample count, and target QPS ...** consistent unless the experiment explicitly requires otherwise.
