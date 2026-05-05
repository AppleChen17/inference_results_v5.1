import os
from typing import Optional, Dict, Sequence


import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__file__)

import random


class Dataset:
    def __init__(self,
        total_sample_count=24576,
        perf_count_override=None,
        dataset_path=None,
    ):
        self.dataset_path = dataset_path
        self.load_processed_dataset()
        self.total_sample_count = min(len(self.input_ids), total_sample_count)
        self.perf_count = perf_count_override or self.total_sample_count


    def load_processed_dataset(self):
        if not os.path.isfile(self.dataset_path):
            log.warning("Processed pickle file {} not found. Please check that the path is correct".format(self.dataset_path))

        log.info("Loading dataset...")
        import pandas as pd
        processed_data = pd.read_pickle(self.dataset_path)
        input_tokens = processed_data['tok_input']

        self.input_ids = []
        self.attention_masks = []
        self.query_types = []

        for ids in input_tokens:
            self.input_ids.append(ids)

        self.stop_ids = []
        if 'tok_stop_sequence' in processed_data.columns:
            stop_tokens = processed_data['tok_stop_sequence']
            for ids in stop_tokens:
                self.stop_ids.append(ids)

        log.info("Finished loading dataset.")


    def postProcess(self, out_tokens, input_seq_lens=None, query_id_list=None, sample_index_list=None):
        pass

    def LoadSamplesToRam(self, sample_list):
        pass

    def UnloadSamplesFromRam(self, sample_list):
        pass

    def __del__(self):
        pass
