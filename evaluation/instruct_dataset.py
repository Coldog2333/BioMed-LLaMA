import copy
from dataclasses import dataclass
import torch
import transformers
from torch.utils.data import Dataset
from collections import defaultdict
import json
import random
from pprint import pprint
from prettytable import PrettyTable
from transformers import PreTrainedTokenizer

IGNORE_INDEX = -100


class InstructDataset(Dataset):
    def __init__(
        self,
        input_filename: str,
        n_epoch: int,
        n_sample_per_epoch: int,
        tokenizer: PreTrainedTokenizer
    ):
        super(InstructDataset, self).__init__()
        all_samples = json.load(open(input_filename, 'r', encoding='utf-8'))

        n_sample_per_source = defaultdict(int)
        for sample in all_samples:
            n_sample_per_source[sample['source']] += 1

        print(n_sample_per_source)

        n_epoch_per_source = {
            'usmleqa/train': 3,
            'medmcqa/train': 0.2,
            'pubmedqa/train': 0.1,
            'AlpacaDataCleaned': 0.5,
            'visual-med-alpaca': 0.5,
            'usmleqa/medpalm': 5,
            'medmcqa/medpalm': 5,
            'pubmedqa/medpalm': 5,
            'liveqa/medpalm': 1,
            'medicationqa/medpalm': 1,
            'usmleqa/medpalm-cot': 3,
            'medmcqa/medpalm-cot': 3,
            'pubmedqa/medpalm-cot': 3,
            'mmlu/medpalm-cot': 3,
            'usmleqa/medpalm2-cot': 3,
            'medmcqa/medpalm2-cot': 3,
            'pubmedqa/medpalm2-cot': 3,
            'mmlu/medpalm2-cot': 3,
            'mmlu-cot': 3,
            'codex-cot': 3
        }

        total_sample = int(sum(n_sample_per_source[source] * n_epoch_per_source[source] for source in n_epoch_per_source.keys()))
        selection_prob_per_source = {
            k: n_sample_per_source[k] * n_epoch_per_source[k] / total_sample
            for k, v in n_epoch_per_source.items()
        }

        print(total_sample)
        pprint(selection_prob_per_source)
        pprint({k: (v * n_epoch * n_sample_per_epoch) for k, v in selection_prob_per_source.items()})

        samples_sorted_by_source = defaultdict(list)
        for sample in all_samples:
            samples_sorted_by_source[sample['source']].append(sample)

        self.samples = []
        selected_sources = random.choices(
            population=list(selection_prob_per_source.keys()),
            weights=list(selection_prob_per_source.values()),
            k=int(n_epoch * n_sample_per_epoch)
        )
        for source in selected_sources:
            self.samples.append(random.choice(samples_sorted_by_source[source]))

        self.tokenizer = tokenizer

    def __getitem__(self, index):
        sample = self.samples[index]

        ## make prompt
        source = sample['input']
        target = f"{sample['output']}{self.tokenizer.eos_token}"

        ## prepare inputs
        example = source + target
        tokenized = self.tokenizer(
            [example, source],
            padding=False,
            # padding="max_length",    # only for debug, otherwise it will be quite slow
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_length=True,
        )
        tokenized['length'] = [min(length, self.tokenizer.model_max_length) for length in tokenized['length']]

        input_ids = torch.tensor(tokenized['input_ids'][0], dtype=torch.long)

        labels = copy.deepcopy(input_ids)

        source_len = tokenized['length'][1]
        labels[:source_len] = IGNORE_INDEX

        return dict(
            input_ids=input_ids,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            labels=labels,
        )

    def __len__(self):
        return len(self.samples)


@dataclass
class InstructTemplate:
    template: str

    def instantiate(self, *args, **kwargs) -> str:
        return self.template.format_map(*args, **kwargs)


if __name__ == '__main__':
    tokenizer = transformers.LlamaTokenizer.from_pretrained('forestai/biomed-llama-7b')
    dataset = InstructDataset(
        input_filename='../dataset/sft.json',
        n_epoch=3,
        n_sample_per_epoch=50000,
        tokenizer=tokenizer
    )
    print(dataset.__getitem__(0))
    print(len(dataset))
