import time
from transformers import AutoTokenizer


DEBUG_MODEL_OUTPUT_PATH = 'model_output.txt'
DEBUG_SAMPLES_LATENCY_OUTPUT_PATH = 'samples_latency_data.txt'


class DebugToolkit:

    def __init__(self, harness_config: dict, llm_config: dict):
        self.samples_latency_data = {}
        self.harness_config = harness_config

        self.model_output_path = self.harness_config.get('debug_model_output_path', DEBUG_MODEL_OUTPUT_PATH)
        self.latency_output_path = self.harness_config.get('debug_latency_output_path', DEBUG_SAMPLES_LATENCY_OUTPUT_PATH)
        if self.harness_config['debug_dump_model_output']:
            open(self.model_output_path, 'w').close()
        self.tokenizer = AutoTokenizer.from_pretrained(self._get_model_path(llm_config))

        if self.harness_config['debug_record_sample_latencies']:
            open(self.latency_output_path, 'w').close()


    def dump(self, text_token_ids):
        with open(self.model_output_path, 'a') as file:  
            for token_ids in text_token_ids:
                text = self.tokenizer.decode(token_ids)
                file.write(f'{text}\n\n')


    def record_sample_latencies(self, sample_id, output_token_ids, input_token_count=None):
        sample_data = self.samples_latency_data.get(sample_id)
        if sample_data is None:
            self.samples_latency_data[sample_id] = SampleData(sample_id)
            if input_token_count:
                self.samples_latency_data[sample_id].input_token_count = input_token_count
        else:
            if sample_data.first_token_time is None:
                sample_data.first_token_time = time.perf_counter_ns()
                assert output_token_ids != None, "output_token_ids should not be None when receiving the first token"

            if output_token_ids is None:
                sample_data.last_token_time = time.perf_counter_ns()

                with open(self.latency_output_path, 'a') as file:
                    ttft = sample_data.first_token_time - sample_data.sent_time
                    tpot = (sample_data.last_token_time - sample_data.first_token_time) / sample_data.output_token_count
                    file.write(f"{sample_data.id=} {ttft=:.0f} {tpot=:.0f} isl={sample_data.input_token_count} osl={sample_data.output_token_count}\n")
            else:
                sample_data.output_token_count += len(output_token_ids)

            self.samples_latency_data[sample_id] = sample_data

    def _get_model_path(self, config: dict):
        return config.get('model', config.get('model_path', None))


class SampleData:
    def __init__(self, id):
        self.id = id
        self.sent_time = time.perf_counter_ns()
        self.first_token_time = None
        self.last_token_time = None
        self.input_token_count = None
        self.output_token_count = 0
