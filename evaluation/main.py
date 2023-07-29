#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from dataclasses import dataclass, field
from typing import Optional

import os

import torch
import transformers
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    Trainer,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer
)
from evaluation.mcqa_dataset import (
    UsmleqaDataset,
    MedmcqaDataset,
    PubmedqaDataset,
    InstructUsmleqaDataset,
    InstructMedmcqaDataset,
    InstructPubmedqaDataset
)
from evaluation.instruct_dataset import (
    InstructDataset,
)
from utils.data import DataCollatorForSupervisedDataset
from modeling_utils import safe_save_model_for_hf_trainer, smart_tokenizer_and_embedding_resize


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    dataset_name: str = field(default=None, metadata={"help": "Name of the training data."})
    data_path: str = field(default=None, metadata={"help": "Path of the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    stage: str = field(default="train")
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    # optimizer
    optim: str = field(default="adamw_torch")
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for Adam optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for Adam optimizer"})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})


if __name__ == "__main__":
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    ## model
    model_config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        use_cache=False if training_args.stage == 'train' else True,
        use_memory_efficient_attention=True if training_args.stage == 'train' else False,
    )

    if training_args.stage == 'debug':
        model = transformers.AutoModelForCausalLM.from_pretrained('facebook/opt-125m')
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=model_config,
            torch_dtype=None if training_args.stage == 'train' else torch.float16
        )

    tokenizer = AutoTokenizer.from_pretrained(
    # tokenizer = transformers.LlamaTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right" if training_args.stage == 'train' else "left",
        use_fast=False
    )

    if "llama" in model_args.model_name_or_path.lower():
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict={'pad_token': '[PAD]', 'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': "<unk>"},
            tokenizer=tokenizer,
            model=model,
        )

    ## data
    dataset_collections = {
        'usmleqa': UsmleqaDataset,
        'medmcqa': MedmcqaDataset,
        'pubmedqa': PubmedqaDataset,
        'instruct': InstructDataset,
        'instruct-usmleqa': InstructUsmleqaDataset,
        'instruct-medmcqa': InstructMedmcqaDataset,
        'instruct-pubmedqa': InstructPubmedqaDataset,
    }

    if data_args.dataset_name in ['usmleqa', 'medmcqa', 'pubmedqa', 'instruct-usmleqa', 'instruct-medmcqa', 'instruct-pubmedqa']:
        data_module = dict(
            train_dataset=dataset_collections[data_args.dataset_name](mode='train', tokenizer=tokenizer) if training_args.stage == 'train' else None,
            eval_dataset=dataset_collections[data_args.dataset_name](mode='test', tokenizer=tokenizer),
            data_collator=DataCollatorForSupervisedDataset(
                tokenizer=tokenizer,
                padding_side="right" if training_args.stage == 'train' else "left",
                is_training=True if training_args.stage == 'train' else False,
            ),
        )
    elif data_args.dataset_name in ['instruct']:
        data_module = dict(
            train_dataset=dataset_collections[data_args.dataset_name](
                input_filename='../dataset/sft_train.json',
                n_epoch=training_args.num_train_epochs,
                n_sample_per_epoch=50000,
                tokenizer=tokenizer
            ) if training_args.stage == 'train' else None,
            eval_dataset=None,
            data_collator=DataCollatorForSupervisedDataset(
                tokenizer=tokenizer,
                padding_side="right" if training_args.stage == 'train' else "left",
                is_training=True if training_args.stage == 'train' else False,
            ),
        )
        training_args.num_train_epochs = 1
    else:
        raise NotImplementedError

    ## training
    if training_args.stage == 'train':
        if os.environ['LOCAL_RANK'] == '0':
            print(model_args, data_args, training_args)

        trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
        trainer.train()
        trainer.save_state()
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    else:
        dataloader = DataLoader(
            data_module['eval_dataset'],
            batch_size=4,
            shuffle=False,
            num_workers=4,
            collate_fn=data_module['data_collator'],
        )
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        accuracy, n_count = 0, 0
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            input_texts = [batch['others'][k]['input_text'] for k in range(len(batch['others']))]
            ground_truths = [batch['others'][k]['output_text'] for k in range(len(batch['others']))]

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=128,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True
                )
                output_token_ids = outputs.sequences.detach().cpu().numpy().tolist()
                raw_output_texts = tokenizer.batch_decode(output_token_ids, skip_special_tokens=True)

                for j, raw_output_text in enumerate(raw_output_texts):
                    n_count += 1

                    output_text = raw_output_text[len(input_texts[j]):]
                    target_text = ground_truths[j]

                    if target_text.endswith('</s>'):
                        target_text = target_text[:-4]  # remove </s>

                    output_text = output_text.lstrip('( ').strip(' )')
                    target_text = target_text.lstrip('( ').strip(' )')

                    if data_args.dataset_name in ['instruct-pubmedqa']:
                        target_text = target_text.split(' ')[1]
                        target_text = target_text.capitalize()

                    if output_text == target_text:
                        accuracy += 1
                    else:
                        print('==')
                        print(output_text)
                        print('--')
                        print(target_text)
                        print('==')

                    print(f'Accuracy: {accuracy / n_count:.4f} ({accuracy}/{n_count})')
