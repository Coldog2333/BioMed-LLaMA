import copy
import json
import math
from dataclasses import dataclass
from typing import Sequence, Dict

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import os
import logging
import transformers
from transformers import LlamaTokenizer


IGNORE_INDEX=-100


def get_num_line(filename):
    return int(os.popen(f'wc -l {filename}').read().split()[0])


def preprocess(text, tokenizer):
    tokenized = tokenizer(
        text,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True
    )
    input_ids = tokenized.input_ids[0]
    labels = copy.deepcopy(input_ids)

    return dict(input_ids=input_ids, labels=labels)


class LMDataset(Dataset):
    def __init__(
        self,
        jsonl_file_path: str,
        tokenizer: transformers.PreTrainedTokenizer
    ):
        # 按输入的路径是文件还是目录来处理
        if os.path.isfile(jsonl_file_path):
            self.jsonl_files = [jsonl_file_path]
        elif os.path.isdir(jsonl_file_path):
            self.jsonl_files = []
            for jsonl_file in os.listdir(jsonl_file_path):
                if jsonl_file.endswith('.jsonl'):
                    self.jsonl_files.append(os.path.join(jsonl_file_path, jsonl_file))
        else:
            raise AssertionError(f'[{jsonl_file_path}] is not a file or a directory.')

        self.samples = []
        for jsonl_file in self.jsonl_files:
            samples = open(jsonl_file, 'r').readlines()
            self.samples += samples
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        text = json.loads(self.samples[index])['text']
        text += self.tokenizer.eos_token
        return preprocess(text, self.tokenizer)

    def __len__(self):
        return len(self.samples)


class LMSuperLargeDataset(IterableDataset):
    def __init__(
        self,
        jsonl_file_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
    ):
        super(LMSuperLargeDataset, self).__init__()

        # 按输入的路径是文件还是目录来处理
        if os.path.isfile(jsonl_file_path):
            self.jsonl_files = [jsonl_file_path]
        elif os.path.isdir(jsonl_file_path):
            self.jsonl_files = []
            for jsonl_file in os.listdir(jsonl_file_path):
                if jsonl_file.endswith('.jsonl'):
                    self.jsonl_files.append(os.path.join(jsonl_file_path, jsonl_file))
        else:
            raise AssertionError(f'[{jsonl_file_path}] is not a file or a directory.')

        self.tokenizer = tokenizer

        # get the number of lines with cache
        self.n_lines = []
        for jsonl_file in self.jsonl_files:
            try:
                # logging.warning('Trying to load #line from cache file: [%s]' % (jsonl_file[:-len('jsonl')] + 'txt'))
                logging.warning_once('Trying to load #line from cache file.')
                this_n_lines = int(open(jsonl_file[:-len('jsonl')] + 'txt', 'r').read())
            except:
                this_n_lines = get_num_line(jsonl_file)
                open(jsonl_file[:-len('jsonl')] + 'txt', 'w').write(str(this_n_lines))
            self.n_lines.append(this_n_lines)

    def __iter__(self):
        for i in range(len(self.jsonl_files)):
            with open(self.jsonl_files[i], 'r', encoding='utf-8') as f:
                for line in f:
                    text = json.loads(line)['text']
                    text += self.tokenizer.eos_token
                    yield preprocess(text, self.tokenizer)

    def __len__(self):
        return sum(self.n_lines)


@dataclass
class LMDataCollator:
    tokenizer: LlamaTokenizer

    def __call__(self, batch):
        input_ids, labels = tuple([b[key] for b in batch] for key in ("input_ids", "labels"))

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            # padding="max_length",  # for confirming the memory usage
            max_length=tokenizer.model_max_length,
            truncation=True
        )
        for text in strings
    ]
    input_ids = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        input_ids_lens=input_ids_lens,
    )


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    padding_side: str = "right"
    is_training: bool = True

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))

        if self.padding_side == "left":
            input_ids = [torch.flip(i, dims=[-1]) for i in input_ids]
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            input_ids = torch.flip(input_ids, dims=[-1])
        else:
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )

        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        if not self.is_training:
            others = []
            for instance in instances:
                others.append({
                    'input_text': instance['input_text'],
                    'output_text': instance['output_text'],
                })

            return dict(
                input_ids=input_ids,
                labels=labels,
                attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
                others=others
            )
        else:
            return dict(
                input_ids=input_ids,
                labels=labels,
                attention_mask=input_ids.ne(self.tokenizer.pad_token_id)
            )
