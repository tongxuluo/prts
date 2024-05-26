<div align="center">
    <img src="https://github.com/tongxuluo/prts/blob/main/stacking_poster.png" width="256" height="256">
</div>

# PRTS: PRe-Training by Stacking (LLM-Stacking)

The official implementation for paper **Stacking Your Transformers: A Closer Look at Model Growth for Efficient LLM Pre-Training** 


<p align="center">
<a href="https://github.com/tongxuluo/prts/blob/main/LICENSE">
<img src='https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg'></a>
<img src='https://img.shields.io/badge/python-3.10+-blue.svg'>
<!-- <img src='https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg'> -->
</p>

<p align="center">
🔔 <a href="https://github.com/tongxuluo/prts" target="_blank">Code</a> • 📃 <a href="https://github.com/tongxuluo/prts" target="_blank">Paper</a> • 🤗 <a href="https://huggingface.co/llm-stacking" target="_blank">Model</a> <br>
</p>


## Abstract
Pre-training LLMs is notoriously expensive.
Model growth emerges as a promising approach by leveraging smaller models to accelerate the training of much larger models.
However, the viability of these approaches in efficient pre-training for LLMs remains underexplored.
This work identifies three critical **obstacles**: (*O1*) the lack of comprehensive evaluation, (*O2*) the untested viability for scaling, and (*O3*) the lack of empirical guidelines, which will be addressed one by one.
To tackle *O1*, we summarize existing approaches into four atomic growth operators and systematically evaluate them in a standardized LLMs pre-training setting.
Our findings reveal that a depthwise stacking operator, $G_{\text{stack}}$, exhibits remarkable acceleration in training, leading to decreased loss and improved overall performance on eight standard NLP benchmarks compared to strong baselines.
Motivated by these promising results, we conduct extensive experiments to delve deeper into $G_{\text{stack}}$ to address *O2* and *O3*.
For *O2*, our study shows that $G_{\text{stack}}$ is scalable and consistently performs well, with experiments up to 7B LLMs after growth and 750B tokens.
For example, compared to a conventionally trained 7B model using 300B tokens, our $G_{\text{stack}}$ model converges to the same loss with just 194B tokens, resulting in a 54.6% FLOPs speedup.
We further address *O3* by formalizing equational guidelines for $G_{\text{stack}}$, making it practical in general LLMs pre-training.
We also provide comprehensive ablation studies of $G_{\text{stack}}$ and in-depth discussions.

## Growth Operators
<p align="center">
  <img src="https://github.com/tongxuluo/prts/blob/main/op.png" alt="Image text">
</p>

## Getting Started
```
git clone https://github.com/tongxuluo/prts.git
cd prts
```

### Data Preparation
#### Download Datasets
```
cd /path/to/dataset
git lfs install
git clone https://huggingface.co/datasets/cerebras/SlimPajama-627B
```

#### Tokenize data
```
python scripts/prepare_slimpajama.py --source_path /path/to/SlimPajama --tokenizer_path path/to/llama  --destination_path data/slimpajama --split validation --percentage 1.0
python scripts/prepare_slimpajama.py --source_path /path/to/SlimPajama --tokenizer_path path/to/llama  --destination_path data/slimpajama --split train --percentage 1.0
```

### Pretraining on Slurm Cluster
We formalize a set of guidelines for effectively utilizing the $G_{\text{stack}}$ operator. For growth timing $d$ (tokens):
$$log_{10}(d) = 0.88\log_{10}(N) + \frac{163.27}{\log_{10}(C)} + -5.74$$

For growth factor $g$, we fix $g=4$ is the best.

For Llama Families:

| Model      | N    | D    | ${d}$  | ${g}$ |
|------------|------|------|--------|-------|
| Llama3-8B  | 8B   | 15T  | 6.58B  | 4     |
| Llama2-7B  | 7B   | 2T   | 11.11B | 4     |
| Llama2-13B | 13B  | 2T   | 15.84B | 4     |
| Llama2-70B | 70B  | 2T   | 42.48B | 4     |

#### Pretraining a Small Base Model with $d$ Tokens From Scratch
We only need the first checkpoint (10B tokens).
```
sbatch base_model.sh
```

#### Create PRTS Config
For example, in the case of $G_{\text{stack}}$, please refer to [prts_lit/utils/config.py](https://github.com/tongxuluo/prts/blob/main/prts_lit/utils/config.py) for more details.
```
{
    "src_config_name": "6L2048H",
    "trg_config_name": "24L2048H",
    "src_init_path": "/path/to/your/base_model/check_point_dir/iter-005000-ckpt.pth",
    "stacking_list": [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5],
    "embd_name": "wte",
    "ln_name": "ln_f",
    "head_name": "lm_head",
    "layer_name": "h"
}
```

#### Start Continual Pretraining
```
sbatch g_stack.sh
```

## TODO
- [x] Open source our code -- 2024.5.24 .
- [x] Open source our last checkpoints of main experiments -- 2024.5.29 .
- [ ] Refactor our code to make it more concise -- 2024.7 .


## Acknowledgement
Our code is based on [TinyLlama](https://github.com/jzhang38/TinyLlama), licensed under the Apache-2.0 license.
```
@misc{zhang2024tinyllama,
      title={TinyLlama: An Open-Source Small Language Model}, 
      author={Peiyuan Zhang and Guangtao Zeng and Tianduo Wang and Wei Lu},
      year={2024},
      eprint={2401.02385},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Citation
If you find our paper inspiring and have utilized it in your work, please cite our paper.
```
@article{du2024stacking,
  title={Stacking Your Transformers: A Closer Look at Model Growth for Efficient LLM Pre-Training},
  author={Du, Wenyu and Luo, Tongxu and Qiu, Zihan and Huang, Zeyu and Shen, Yikang and Cheng, Reynold and Guo, Yike and Fu, Jie},
  journal={arXiv preprint},
  year={2024}
}
```
