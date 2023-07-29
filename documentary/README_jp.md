# BioMed-LLaMA: Continuous Pretraining LLaMA with Biomedical Abstracts and Papers

[Junfeng Jiang](https://coldog2333.github.io/)<sup>1</sup>, Qiang Zhang<sup>2</sup>, Akiko Aizawa<sup>1</sup>, Renjing Xu<sup>2</sup>

[University of Tokyo](https://www.i.u-tokyo.ac.jp/index_e.shtml)<sup>1</sup>    [The Hong Kong University of Science and Technology](https://hkust.edu.hk/)<sup>2</sup>

<h4 align="center">
    <p>
        <b>English</b> |
        <a href="https://github.com/OptimalScale/LMFlow/blob/main/readme/README_zh.md">简体中文</a> |
        <a href="https://github.com/OptimalScale/LMFlow/blob/main/readme/README_jp.md">日本語</a> |
    <p>
</h4>
## Introduction

BioMed-LLaMA-7b is a large language model (LLM) having 7 billion parameters pretrained continuously from MetaAI's LLaMA-7b checkpoint on biomedical abstracts and papers from The Pile, namely, the PubMed-abstract and PubMed-central subsets.

In this repository, we also provide the codes for continuous pretraining, finetuning, and evaluation. Hope that this work can be beneficial to the biomedical NLP community.

## Pretraining resources

[The Pile](http://pile.eleuther.ai/) is a large-scale high-quality dataset of diverse text sources that is designed to be used for pretraining large language models. It contains 825 GiB of text from 22 diverse sources, including Wikipedia, PubMed abstracts, PubMed Central papers, etc. We extracted the **PubMed-abstract** and **PubMed-central** subsets from The Pile as our pretraining resources, which contain approximately 30M abstracts and 5M papers.

After extraction, we obtained 213 GiB of text containing about 63B tokens. We trained the LLaMA-7b model on these data for 1 epoch to avoid overfitting to the pretraining data.

## Training Procedure

Since it is a continuous pretraining, we mainly follow the hyperparameters of LLaMA-7b as shown below.

|                   |               |
| ------------------- | --------------- |
| max_seq_length    | 2048          |
| lr                | 3e-5          |
| batch size        | 2048          |
| betas             | \[0.9, 0.95\] |
| weight decay      | 0.1           |
| gradient clipping | 1.0           |

The model was trained on an 8-node HPC cluster containing 32 NVIDIA A100-80GB GPUs in total lasting about a week.

We conducted several optimization strategies to speed up training and reduce memory consumption.

+ We used [PyTorch FSDP](https://pytorch.org/docs/stable/fsdp.html) to enable model parallelism. However, since the network bandwidth across nodes in our cluster is limited, we adopted **hybrid sharing** strategy to reduce node-wise communication cost.
+ Gradient accumulation is also applied to reduce GPU-wise communication cost.
+ We also used [xformers](https://github.com/facebookresearch/xformers) to conduct effective attention computation to reduce memory consumption and speed up training.
+ Mixed precision training (bf16+tf32) is also used to reduce memory consumption and speed up training. Though the data type of LLaMA's model weights is float16, we didn't observe any difference between fp16 and bf16 training in our preliminary experiments.

### Training Loss Curve

Here below is the curve of training loss, where running average smoothing is applied for visualization.

<div align="center">  
  <img src="./documentary/biomed-llama-7b_training_curve.png" width = "505" height = "345" alt="Training Loss Curve" align=center />
</div>

## Evaluation

We conducted comparison mainly with vanilla LLaMA-7B, [PMC-LLaMA](https://github.com/chaoyi-wu/PMC-LLaMA), and [BioMedLM](https://github.com/stanford-crfm/BioMedLM). Some other models are also included for some of the downstream tasks. Evaluating language models on some downstream tasks (e.g., QA) is not trivial since they tend to generate free-form answers. Therefore, we show the *potential accuracy* of them by computing the perplexity of each option given the question (and the abstract for PubMedQA) using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). The option with lowest perplexity is chosen as the final answer. 

Since MedQA and MedMCQA are not implemented by EleutherAI, we implemented them by ourselves. So please use the version of lm-evaluation-harness in this repository to evaluate them.

Note that BioMedLM was trained on the same pretraining resources but more epochs (6 epochs in total containing 300B tokens), and PMC-LLaMA-7B was trained on 4.8M PubMedCentral papers for 5 epochs.

| Model           | Strategy | PubMed-A | PubMed-C | USMLE (4/5)       | MedMCQA    | PubMedQA   |
| ----------------- | ---- | ---------------------- | --------------------- | ------------------------- | ------------ | ------------ |
| Random          | -  | -                    | -                   | 0.25 / 0.5                       | 0.25          | 0.33        |
| GPT-Neo (2.7B)  | 0-shot | 19.1207              | 20.8701             | 0.2781 / 0.2412         | 0.2570     | 0.5640     |
| BioMedLM (2.7B) | 0-shot | **15.6959**          | **18.6799**         | 0.2993 / 0.2624         | 0.2744     | 0.5520     |
| LLaMA-7B        | 0-shot | 20.1107              | 29.0583             | 0.3339 / 0.2742         | **0.2933** | **0.7520** |
| PMC-LLaMA-7B    | 0-shot | 36.8191              | 39.5381             | 0.3441 / 0.2883         | 0.2850     | 0.6640     |
| BioMed-LLaMA-7B | 0-shot | 15.7774              | 20.9322             | **0.3535** / **0.3032** | 0.2921     | 0.6160     |
| LLaMA-7B | few-shot | -              | -             | 0.3661 (3) / 0.3174(3) | 0.2991 (10) | **0.713** (1) |
| BioMed-LLaMA-7B | few-shot | -              | -             | **0.3668** (3) / **0.3229** (3)         | **0.3007** (10)     | 0.702 (1)     |
| LLaMA-7B | fine-tune | -              | -             | unstable | 0.4994 | **0.764** |
| BioMed-LLaMA-7B | fine-tune | -              | -             | unstable         |  **0.5357**    | 0.763     |

*PubMed-A: Pile/PubMed-Abstracts, PubMed-C: Pile/PubMed-Central, USMLE: MedQA-USMLEQA

## Instruction Tuning

Existing commercial LLMs achieve an excellent performance on medical tasks like USMLE-QA, especially when performing few-shot inference. However, they usually have tremendous number of parameters, so the inference requires many computation resources and time, especially when adding few-shot demonstrations to the inputting prompt. Finetuning on these demonstrations is also impossible. However, our model is quite smaller and we have many downstream tasks to be evaluated, so we conducted instruction tuning with these few-shot examples instead of performing in-context prompting.

We collected diverse instruction tuning data from various resources:

| Source                                                                 | #Sample | MixtureP | Domain  |
| ------------------------------------------------------------------------ | --------- | ------------ | --------- |
| MedQA-USMLE/train                                                        | 10178   | 21.45%    | Medical |
| [MedMCQA/train](https://huggingface.co/datasets/medmcqa)             | 182822   | 25.69%    | Medical |
| [PubMedQA/train](https://huggingface.co/datasets/pubmed_qa)         | 211269   | 14.84%    | Medical |
| [AlpacaDataCleaned](https://github.com/gururise/AlpacaDataCleaned)     | 51760   | 18.18%   | Open    |
| [visual-med-alpaca](https://github.com/cambridgeltl/visual-med-alpaca) | 54412   | 19.11%   | Medical |
| [medpalm](https://arxiv.org/pdf/2212.13138.pdf)                       | 24       | 0.05%    | Medical |
| [medpalm-cot](https://arxiv.org/pdf/2212.13138.pdf)                   | 19       | 0.04%    | Medical |
| [medpalm2-cot](https://arxiv.org/pdf/2305.09617.pdf)                   | 19       | 0.04%    | Medical |
| [mmlu-cot](https://github.com/jasonwei20/flan-2)                       | 282     | 0.6%    | Science |
| [codex-cot](https://arxiv.org/pdf/2207.08143v3.pdf)                    | 3       | 0.006%    | Medical |


After instruction tuning, we can find that BioMed-LLaMA can benefit more than vanilla LLaMA from the instruction tuning, especially on MedQA-USMLE. However, the performances on MedMCQA and PubMedQA are not improved comparing to finetuning. We think that there are three possible reasons:
1. During instruction tuning, even though we have a large number of training samples for MedMCQA and PubMedQA, these data only contain part of the original training data. So the models may not be able to learn the full distribution of the training data, and therefore perform worse than finetuning with the whole training datasets.
2. The questions of MedMCQA are quite short, whereas other instruction tuning data generally has longer input.
3. The answers of PubMedQA are quite short (Yes/No/Maybe), making them more difficult to optimize during jointly training.


| Model           | Strategy | USMLE (4)       | MedMCQA    | PubMedQA   |
| ----------------- | ---- |  ------------------------- | ------------ | ------------ |
| LLaMA-7B | instructed |  0.4391 | 0.4236 | 0.744 |
| BioMed-LLaMA-7B | instructed |  **0.487** | **0.4475** | **0.757**


## Citation
Please cite this repo if you find the codes or contents are useful for your research.
```
@misc{alpaca,
  author = {Junfeng Jiang, Qiang Zhang, Akiko Aizawa, and Renjing Xu},
  title = {BioMed-LLaMA: Continuous Pretraining LLaMA with Biomedical Abstracts and Papers},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Coldog2333/BioMed-LLaMA}},
}
```
