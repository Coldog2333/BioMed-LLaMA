# @Author: Coldog

"""Pretraining BioMed-LLaMA with LLaMA"""
import os
from dataclasses import dataclass, field
from typing import Optional, Dict
import transformers
from transformers import Trainer

from modeling_utils import safe_save_model_for_hf_trainer, smart_tokenizer_and_embedding_resize
from utils.data import LMSuperLargeDataset, LMDataCollator


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    eval_data_path: str = field(default=None, metadata={"help": "Path to the evaluation data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    # optimizer
    optim: str = field(default="adamw_torch")
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for Adam optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for Adam optimizer"})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})


def make_language_modeling_data_module(data_args, tokenizer):
    ## dataset
    # If you have a very large RAM:
    # dataset = LMDataset(data_args.data_path, tokenizer)
    # If you don't have a very large RAM:
    train_dataset = LMSuperLargeDataset(data_args.data_path, tokenizer)

    if data_args.eval_data_path is not None:
        eval_dataset = LMSuperLargeDataset(data_args.eval_data_path, tokenizer)
    else:
        eval_dataset = None

    if os.environ['LOCAL_RANK'] == '0':
        print(f"train_dataset: {len(train_dataset)}")

    data_collator = LMDataCollator(tokenizer=tokenizer)

    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )


if __name__ == '__main__':
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if os.environ['LOCAL_RANK'] == '0':
        print(model_args, data_args, training_args)

    ## model
    model_config = transformers.LlamaConfig.from_pretrained(
        model_args.model_name_or_path,
        use_cache=False,
        use_memory_efficient_attention=True,
    )

    model = transformers.LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        config=model_config
    )

    ## tokenizer
    tokenizer = transformers.LlamaTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token={"pad_token": "<pad>"}),
            tokenizer=tokenizer,
            model=model,
        )
    if "llama" in model_args.model_name_or_path:
        tokenizer.add_special_tokens(
            {
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "pad_token": "<pad>"
            }
        )

    data_module = make_language_modeling_data_module(data_args, tokenizer)
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module
    )

    if os.path.exists(training_args.output_dir) and os.listdir(training_args.output_dir):
        resume_from_checkpoint = True
        print('Resuming from checkpoint: [%s] ...' % training_args.output_dir)
    else:
        resume_from_checkpoint = False
        print('Training from scratch...')

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_state()

    try:
        print('Saving model from transformers.trainer...')
        trainer.save_model(output_dir=training_args.output_dir)
        print('Done.')
    except:
        pass

    try:
        print('Saving model from customized saver...')
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=os.path.join(training_args.output_dir, 'hf_model'))
        print('Done.')
    except:
        pass
