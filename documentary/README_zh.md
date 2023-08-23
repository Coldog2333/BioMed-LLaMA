# BioMed-LLaMA: Continuous Pretraining LLaMA with Biomedical Abstracts and Papers

[Junfeng Jiang](https://coldog2333.github.io/)<sup>1</sup>, Qiang Zhang<sup>2</sup>, Akiko Aizawa<sup>1</sup>, Renjing Xu<sup>2</sup>

[University of Tokyo](https://www.i.u-tokyo.ac.jp/index_e.shtml)<sup>1</sup>    [The Hong Kong University of Science and Technology](https://hkust.edu.hk/)<sup>2</sup>

<h4 align="center">
    <p>
        <a href="https://github.com/Coldog2333/BioMed-LLaMA/blob/master/README.md">English</a> |
        <b>简体中文</b> |
        <a href="https://github.com/Coldog2333/BioMed-LLaMA/blob/master/documentary/README_jp.md">日本語</a> |
    <p>
</h4>

## 介绍

BioMed-LLaMA-7B是一个大型语言模型（LLM），它有70亿参数，是在MetaAI的LLaMA-7B模型的基础上，使用[The Pile](http://pile.eleuther.ai/)中的生物医学论文的摘要和全文（即PubMed-abstract和PubMed-central子集）进行继续预训练得到的。

在本仓库中，我们还提供了用于继续预训练、微调和评估的代码。希望这项工作能够对生物医学相关的NLP社区有所帮助。

## 预训练数据

The Pile是一个大规模的、高质量的、多样化文本数据集，旨在用于预训练大型语言模型。它包含来自22个不同来源的825 GiB文本，包括维基百科、PubMed论文摘要、PubMed Central论文等。我们从The Pile中提取了**PubMed-abstract**和**PubMed-central**子集作为我们的预训练数据，其中包含大约3000万篇论文的摘要以及500万篇论文的文本数据。

我们最终提取了213 GiB的文本，其中包含大约630亿个token。我们在这些数据上对LLaMA-7B模型进行了1个epoch的训练。这样做也可以避免过拟合预训练数据（因为大模型的拟合能力非常强）。

## 训练过程

由于这是在做继续预训练，所以我们主要遵循LLaMA-7B的超参数设置（如下所示）：

|                   |               |
| ------------------- | --------------- |
| max_seq_length    | 2048          |
| lr                | 3e-5          |
| batch size        | 2048          |
| betas             | \[0.9, 0.95\] |
| weight decay      | 0.1           |
| gradient clipping | 1.0           |

本模型用了HPC集群中8个节点来进行训练，总共32个NVIDIA A100-80GB GPU，总训练时长大约为一周。

我们采用了多个优化策略来加速训练和减少GPU显存的消耗。

+ 我们使用了[PyTorch FSDP](https://pytorch.org/docs/stable/fsdp.html)来实现模型并行。但是，由于我们集群中节点间的网络带宽有限（≈13GB/s），我们采用了**Hybrid Sharding**的策略来减少节点间的通信开销。如果需要用到此功能，可以安装[Coldog2333/transformers](https://github.com/Coldog2333/transformers)库（基于transformers v4.28.1）。
+ 我们使用了梯度累积来减少节点内GPU之间的通信开销。
+ 我们使用了[xformers](https://github.com/facebookresearch/xformers)来进行高效的注意力计算，以减少显存消耗并加速训练。
+ 我们还使用了混合精度训练（bf16+tf32）来减少显存消耗并加速训练。虽然LLaMA模型权重的数据类型是float16，但我们在初步实验中没有观察到fp16混合精度训练和bf16混合精度训练之间的差异。

### 训练损失曲线

下面是训练时的损失曲线。为了更好地展示，我们对损失进行了平滑处理。

<div align="center">  
  <img src="./documentary/biomed-llama-7b_training_curve.png" width = "505" height = "345" alt="训练损失曲线" align=center />
</div>

## 评估

我们主要围绕原生LLaMA-7B，[PMC-LLaMA](https://github.com/chaoyi-wu/PMC-LLaMA)，和[BioMedLM](https://github.com/stanford-crfm/BioMedLM)进行比较。其它的一些相关模型也被考虑在内，但是只在某些下游任务上进行评估。在下游任务（比如问答任务）上评估语言模型并不是一件简单的事情，因为语言模型通常倾向于生成自由形式的答案。所以，我们用[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)库计算一个*潜在正确率*来作为评估指标。即计算给定问题（在PubMedQA中还会输入摘要）的情况下，每个选项的困惑度（perplexity）。困惑度最低的选项被选为最终答案。

由于EleutherAI并没有实现MedQA和MedMCQA，所以如果需要在这两个任务上进行评估，请使用**本仓库中的**lm-evaluation-harness版本。



Note that BioMedLM was trained on the same pretraining resources but more epochs (6 epochs in total containing 300B tokens), and PMC-LLaMA-7B was trained on 4.8M PubMedCentral papers for 5 epochs.

注意，虽然BioMedLM的预训练数据与我们的BioMed-LLaMA相同，但是它训练了更多的epoch（总共6个epoch，包含300B个token）；而PMC-LLaMA-7B是在480万篇PubMedCentral论文上进行了5个epoch的训练。


| 模型           | 评估方法 | PubMed-A | PubMed-C | USMLE (4/5)       | MedMCQA    | PubMedQA   |
| ----------------- | ----------- | ------------- | ------------- | --------------------------------- | ----------------- | --------------- |
| Random          | -         | -           | -           | 0.25 / 0.5                      | 0.25            | 0.33          |
| GPT-Neo (2.7B)  | 0-shot    | 19.1207     | 20.8701     | 0.2781 / 0.2412                 | 0.2570          | 0.5640        |
| BioMedLM (2.7B) | 0-shot    | **15.6959** | **18.6799** | 0.2993 / 0.2624                 | 0.2744          | 0.5520        |
| LLaMA-7B        | 0-shot    | 20.1107     | 29.0583     | 0.3339 / 0.2742                 | **0.2933**      | **0.7520**    |
| PMC-LLaMA-7B    | 0-shot    | 36.8191     | 39.5381     | 0.3441 / 0.2883                 | 0.2850          | 0.6640        |
| BioMed-LLaMA-7B | 0-shot    | 15.7774     | 20.9322     | **0.3535** / **0.3032**         | 0.2921          | 0.6160        |
| LLaMA-7B        | few-shot  | -           | -           | 0.3661 (3) / 0.3174(3)          | 0.2991 (10)     | **0.713** (1) |
| BioMed-LLaMA-7B | few-shot  | -           | -           | **0.3668** (3) / **0.3229** (3) | **0.3007** (10) | 0.702 (1)     |
| LLaMA-7B        | fine-tune | -           | -           | 0.3946±0.008                   | 0.4994          | **0.764**     |
| BioMed-LLaMA-7B | fine-tune | -           | -           | 0.4072±0.012                   | **0.5357**      | 0.763         |

*PubMed-A: Pile/PubMed-Abstracts, PubMed-C: Pile/PubMed-Central, USMLE: MedQA-USMLEQA

## 指令微调

Existing commercial LLMs achieve an excellent performance on medical tasks like USMLE-QA, especially when performing few-shot inference. However, they usually have tremendous number of parameters, so the inference requires many computation resources and time, especially when adding few-shot demonstrations to the inputting prompt. Finetuning on these demonstrations is also impossible. However, our model is quite smaller and we have many downstream tasks to be evaluated, so we conducted instruction tuning with these few-shot examples instead of performing in-context prompting.

现存的商用语言模型在医学任务上（比如USMLE-QA）表现出了很好的性能，尤其是结合few-shot例子进行推理。然而，它们通常有非常多的参数，所以推理需要很多的计算资源和时间，尤其是在输入提示时添加少样本示例，并且要是在这样的大模型上去对这些示例进行微调也是不太现实的（虽然他们确实也会用少量生物医学领域的数据进行全参数微调）。然而，我们的模型非常小，而且我们有很多下游任务需要评估，在计算资源受限的前提下，我们选择使用这些少样本示例进行指令微调，而不进行in-context learning.

我们从各个地方收集了指令微调数据，具体如下：

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


经过指令微调后，我们发现相对于LLaMA而言，BioMed-LLaMA从指令微调中得到的收益更多，尤其是在MedQA-USMLE上。然而，相比于传统的finetuning，指令微调之后的模型在MedMCQA和PubMedQA上的表现并没有提升得那么多。我们认为有三个可能的原因：
1. 在指令微调过程中，尽管MedMCQA和PubMedQA的训练样本已经有很多了，但是这些数据只包含原始训练数据的一部分。因此，模型可能无法学习到整个训练数据的分布，从而导致指令微调后的表现不如在整个训练数据上进行finetuning。
2. MedMCQA的问题非常短，而其他指令微调数据通常有更长的输入。
3. PubMedQA的答案非常短（Yes/No/Maybe），这使得在联合训练过程中更难优化。


| Model           | Strategy   | USMLE (4) | MedMCQA    | PubMedQA  |
| ----------------- | ------------ | ----------- | ------------ | ----------- |
| LLaMA-7B        | instructed | 0.4391    | 0.4236     | 0.744     |
| BioMed-LLaMA-7B | instructed | **0.487** | **0.4475** | **0.757** |

## 致谢
感谢香港科技大学和JST SPRING（次世代研究者挑戦的研究プログラム）在计算资源和资金上的支持。感谢MetaAI分享Llama模型。感谢其他研究人员分享他们的数据和代码。
另外特别感谢[@anchen1011](https://github.com/anchen1011) 对本研究提供的宝贵建议。

## 引用
如果本仓库的代码或内容对你的研究有帮助，请引用本仓库。
```
@misc{biomedllama,
  author = {Junfeng Jiang, Qiang Zhang, Akiko Aizawa, and Renjing Xu},
  title = {BioMed-LLaMA: Continuous Pretraining LLaMA with Biomedical Abstracts and Papers},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Coldog2333/BioMed-LLaMA}},
}
```

```bibtex
@article{touvron2023llama,
  title={LLaMA: Open and Efficient Foundation Language Models},
  author={Touvron, Hugo and Lavril, Thibaut and Izacard, Gautier and Martinet, Xavier and Lachaux, Marie-Anne and Lacroix, Timoth{\'e}e and Rozi{\`e}re, Baptiste and Goyal, Naman and Hambro, Eric and Azhar, Faisal and Rodriguez, Aurelien and Joulin, Armand and Grave, Edouard and Lample, Guillaume},
  journal={arXiv preprint arXiv:2302.13971},
  year={2023}
}
```
