import copy
from dataclasses import dataclass
from typing import List
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from evaluation.instruct_dataset import InstructTemplate

IGNORE_INDEX = -100


@dataclass
class MCQASample:
    question: str
    options: List[str]
    answer: str
    context: str = None
    explanation: str = None
    sample_id: str = None

    def __post_init__(self):
        self.answer_idx = 'ABCDE'[self.options.index(self.answer)]
        self.n_option = len(self.options)


def _process_usmleqa(original_sample, source):
    # 5-options: https://huggingface.co/datasets/bigbio/med_qa/viewer/med_qa_en_bigbio_qa/train?row=0
    # 4-options: https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options
    question = original_sample['question']
    answer = original_sample['answer']

    if source == 'bigbio':
        assert len(original_sample['answer']) == 1
        answer = answer[0]

        options = original_sample['choices']
        sample_id = original_sample['id']
    elif source == 'gbaker':
        options = [original_sample['options'][key] for key in 'ABCD']
        sample_id = None
    else:
        raise ValueError(f'Unknown source: {source}')

    return MCQASample(
        question=question,
        options=options,
        answer=answer,
        sample_id=sample_id,
    )


def _process_medmcqa(original_sample):
    # https://huggingface.co/datasets/medmcqa
    question = original_sample['question']
    options = [original_sample[key] for key in ['opa', 'opb', 'opc', 'opd']]
    answer = options[original_sample['cop']]

    explanation = original_sample['exp']
    sample_id = original_sample['id']

    return MCQASample(
        question=question,
        options=options,
        answer=answer,
        explanation=explanation,
        sample_id=sample_id,
    )


def _process_pubmedqa(original_sample):
    # pubmed_qa
    # ref: test->pqa_labeled, train->pqa_artificial
    # https://huggingface.co/datasets/pubmed_qa/viewer/pqa_labeled/train
    question = original_sample['question']
    options = ['yes', 'no', 'maybe']
    answer = original_sample['final_decision']

    context = "\n".join(original_sample['context']['contexts'])
    sample_id = original_sample['pubid']

    return MCQASample(
        question=question,
        options=options,
        answer=answer,
        context=context,
        sample_id=sample_id,
    )


class MCQADataset(Dataset):
    def __init__(self, dataset_path, dataset_name, mode, dataset_process_fn, tokenizer):
        """
        :param dataset_path:        The path to the huggingface dataset
        :param dataset_name:        The sub-dataset name
        :param mode:                Options: train, validation, test
        :param dataset_process_fn:  How to process the sample
        :param tokenizer:           Tokenizer
        """
        super(MCQADataset, self).__init__()
        self.mode = mode
        self.tokenizer = tokenizer

        dataset = load_dataset(dataset_path, dataset_name)
        self.samples = list(map(dataset_process_fn, dataset[mode]))

    def __getitem__(self, index):
        sample = self.samples[index]

        ## make prompt
        source = ''
        if sample.context:
            source += '### Context: ' + sample.context + '\n'

        source += '### Question: ' + sample.question + '\n'

        option_text = ' '.join([f'({chr(ord("A") + i)}) {option}' for i, option in enumerate(sample.options)])
        source += '### Options: ' + option_text + '\n'

        source += '### Answer:'

        target = f' {sample.answer}{self.tokenizer.eos_token}'

        ## prepare inputs
        example = source + target
        tokenized = self.tokenizer(
            [example, source],
            padding=False,
            # padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_length=True,
        )
        tokenized['length'] = [min(length, self.tokenizer.model_max_length) for length in tokenized['length']]

        input_ids = torch.tensor(tokenized['input_ids'][0], dtype=torch.long)

        source_len = tokenized['length'][1]
        labels = copy.deepcopy(input_ids)
        labels[:source_len] = IGNORE_INDEX

        return dict(
            input_ids=input_ids,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            labels=labels,
            input_text=source,
            output_text=target
        )

    def __len__(self):
        return len(self.samples)


class UsmleqaDataset(MCQADataset):
    def __init__(self, mode, tokenizer, source='gbaker'):
        if source == 'bigbio':
            dataset_path, dataset_name = 'bigbio/med_qa', 'med_qa_en_bigbio_qa'
            dataset_process_fn = lambda original_sample: _process_usmleqa(original_sample, source='bigbio')
        elif source == 'gbaker':
            dataset_path, dataset_name = 'GBaker/MedQA-USMLE-4-options', None
            dataset_process_fn = lambda original_sample: _process_usmleqa(original_sample, source='gbaker')
        else:
            raise ValueError(f'Unknown source: {source}')

        super(UsmleqaDataset, self).__init__(
            dataset_path,
            dataset_name,
            mode=mode,
            dataset_process_fn=dataset_process_fn,
            tokenizer=tokenizer
        )


class MedmcqaDataset(MCQADataset):
    def __init__(self, mode, tokenizer):
        super(MedmcqaDataset, self).__init__(
            dataset_path='medmcqa',
            dataset_name=None,
            mode=mode,
            dataset_process_fn=_process_medmcqa,
            tokenizer=tokenizer
        )


class PubmedqaDataset(MCQADataset):
    def __init__(self, mode, tokenizer):
        if mode == 'train':
            dataset_name = 'pqa_artificial'
        else:
            dataset_name = 'pqa_labeled'

        super(PubmedqaDataset, self).__init__(
            dataset_path='pubmed_qa',
            dataset_name=dataset_name,
            mode='train',
            dataset_process_fn=_process_pubmedqa,
            tokenizer=tokenizer
        )


class InstructMCDataset(MCQADataset):
    def __init__(
        self,
        dataset_path,
        dataset_name,
        mode,
        dataset_process_fn,
        tokenizer
    ):
        super(InstructMCDataset, self).__init__(
            dataset_path,
            dataset_name,
            mode,
            dataset_process_fn,
            tokenizer
        )
        self.template = self.__template__()

    def __template__(self):
        raise NotImplementedError

    def get_source(self, sample):
        raise NotImplementedError

    def __getitem__(self, index):
        sample = self.samples[index]
        source = self.get_source(sample)
        target = f"({sample.answer_idx}) {sample.answer}{self.tokenizer.eos_token}"
        # target = f"{sample.answer}{self.tokenizer.eos_token}"

        ## prepare inputs
        tokenized = self.tokenizer(
            [source, target],
            padding=False,
            # padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_length=True,
        )
        input_ids = torch.tensor(tokenized['input_ids'][0], dtype=torch.long)
        labels = torch.tensor(tokenized['input_ids'][1], dtype=torch.long)

        return dict(
            input_ids=input_ids,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            labels=labels,
            input_text=source,
            output_text=target
        )


class InstructUsmleqaDataset(InstructMCDataset):
    def __init__(self, mode, tokenizer, source='gbaker'):
        if source == 'bigbio':
            dataset_path, dataset_name = 'bigbio/med_qa', 'med_qa_en_bigbio_qa'
            dataset_process_fn = lambda original_sample: _process_usmleqa(original_sample, source='bigbio')
        elif source == 'gbaker':
            dataset_path, dataset_name = 'GBaker/MedQA-USMLE-4-options', None
            dataset_process_fn = lambda original_sample: _process_usmleqa(original_sample, source='gbaker')
        else:
            raise ValueError(f'Unknown source: {source}')

        super(InstructUsmleqaDataset, self).__init__(
            dataset_path,
            dataset_name,
            mode=mode,
            dataset_process_fn=dataset_process_fn,
            tokenizer=tokenizer
        )
        self.template = self.__template__()

    def __template__(self):
        return InstructTemplate(
            template=(
                "The following is a multiple-choice question (with options) about medical knowledge. "
                "Please select the most appropriate one from the following answer candidates as the final answer.\n\n"
                "### Question: {question}\n\n"
                "### Options:\n{options}\n\n"
                "### Answer:("
            )
        )

    def get_source(self, sample):
        return self.template.instantiate(
            {
                'question': sample.question,
                'options': '\n'.join([f"({option_id}) {option}" for option_id, option in zip('ABCD', sample.options)])
            }
        )


class InstructMedmcqaDataset(InstructUsmleqaDataset, InstructMCDataset):
    def __init__(self, mode, tokenizer):
        InstructMCDataset.__init__(
            self,
            dataset_path="medmcqa",
            dataset_name=None,
            mode=mode,
            dataset_process_fn=_process_medmcqa,
            tokenizer=tokenizer
        )
        self.template = self.__template__()


class InstructPubmedqaDataset(InstructMCDataset):
    def __init__(self, mode, tokenizer):
        if mode == 'train':
            dataset_name = 'pqa_artificial'
        else:
            dataset_name = 'pqa_labeled'

        super(InstructPubmedqaDataset, self).__init__(
            dataset_path="pubmed_qa",
            dataset_name=dataset_name,
            mode='train',
            dataset_process_fn=_process_pubmedqa,
            tokenizer=tokenizer
        )

    def __template__(self):
        return InstructTemplate(
            template=(
                "The following is a medical question paired with an abstract from PubMed. "
                "Please answer Yes / No / Maybe to the question given the scientific evidence as the final answer.\n\n"
                "### Abstract: {abstract}\n\n"
                "### Question: {question}\n\n"
                "### Answer:"
            )
        )

    def get_source(self, sample):
        return self.template.instantiate(
            {
                'abstract': sample.context,
                'question': sample.question,
            }
        )


if __name__ == '__main__':
    from transformers import LlamaTokenizer

    tokenizer = LlamaTokenizer.from_pretrained('openlm-research/open_llama_3b')

    dataset = UsmleqaDataset('train', tokenizer, source='gbaker')
    print(len(dataset))

    dataset = UsmleqaDataset('train', tokenizer, source='bigbio')
    print(len(dataset))

    dataset = MedmcqaDataset('train', tokenizer)
    print(len(dataset))

    dataset = PubmedqaDataset('train', tokenizer)
    print(len(dataset))

