# PRTS: PRe-Training by Stacking

official implementation for paper **Stacking Your Transformers: A Closer Look at Model Growth for Efficient LLM Pre-Training** 


<p align="center">
<a href="https://github.com/Kuroxiro/prts/blob/main/LICENSE">
<img src='https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg'></a>
<img src='https://img.shields.io/badge/python-3.10+-blue.svg'>
<!-- <img src='https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg'> -->
</p>

<p align="center">
🔔 <a href="https://github.com/tongxuluo/prts" target="_blank">Code</a> • 📃 <a href="https://github.com/tongxuluo/prts" target="_blank">Paper</a> • 🤗 <a href="https://huggingface.co/datasets/cerebras/SlimPajama-627B" target="_blank">Model</a> <br>
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
  <img src="https://github.com/Kuroxiro/prts/blob/main/op.png" alt="Image text">
</p>

## Getting Started
```
git clone https://github.com/Kuroxiro/prts.git
cd prts
```
