<a id="top"></a>
<div align="center">
  <img src="./assets/logo.png" width="500"> 
  
  <h1>(ICASSP 2026) HINT: Composed Image Retrieval with Dual-Path Compositional Contextualized Network</h1>
  
  <p>
      <a href="https://2026.ieeeicassp.org/"><img src="https://img.shields.io/badge/ICASSP-2026-blue.svg?style=flat-square" alt="ICASSP 2026"></a>
      <a href="https://arxiv.org/abs/coming soon"><img alt='arXiv' src="https://img.shields.io/badge/arXiv-Coming.Soon-b31b1b.svg"></a>
      <a href="[TODO: 论文正式PDF链接]"><img alt='Paper' src="https://img.shields.io/badge/Paper-ICASSP-green.svg"></a>
    <a href="https://zh-mingyu.github.id/HINT.github.io"><img alt='page' src="https://img.shields.io/badge/Website-orange"></a>
    <a href="https://zh-mingyu.github.id"><img src="https://img.shields.io/badge/Author/Page-blue.svg" alt="Author Page"></a>
    <a href="https://pytorch.org/get-started/locally"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-EE4C2C?&logo=pytorch&logoColor=white"></a>
    <img src="https://img.shields.io/badge/python-≥3.8-blue?style=flat-square" alt="Python">
    <a href="https://github.com/"><img alt='stars' src="https://img.shields.io/github/stars/zh-mingyu/HINT?style=social"></a>
  </p>

  <p>
    <b>Accepted by ICASSP 2026:</b> A novel contextualized network tackling the neglect of contextual information in Composed Image Retrieval (CIR) by amplifying similarity differences between matching and non-matching samples.
  </p>
</div>


## 📌 Introduction

**HINT** (dual-patH composItional coNtextualized neTwork) is our proposed framework for Composed Image Retrieval (CIR), accepted by ICASSP 2026. Although existing methods have made significant progress, they often neglect contextual information in discriminating matching samples. To address the implicit dependencies and the lack of a differential amplification mechanism, HINT systematically models contextual structure to improve the upper performance of CIR models in complex scenarios.

[⬆ Back to top](#top)

## 📢 News
* **[2026-03-26]** 🚀 Initial setup for the HINT repository. Source code is scheduled for release in April 2026.
* **[2026-01-18]** 🔥 Our paper *"HINT: COMPOSED IMAGE RETRIEVAL WITH DUAL-PATH COMPOSITIONAL CONTEXTUALIZED NETWORK"* has been accepted by **ICASSP 2026**!

[⬆ Back to top](#top)

## ✨ Key Features

  - 🧠 **Dual Context Extraction (DCE)**: Extracts both intra-modal context and cross-modal context, enhancing joint semantic representation by integrating multimodal contextual information.
  - 📏 **Quantification of Contextual Relevance (QCR)**: Evaluates the relevance between cross-modal contextual information and the target image semantics, enabling the quantification of implicit dependencies.
  - 🛡️ **Dual-Path Consistency Constraints (DPCC)**: Optimizes the training process by constraining the representation consistency between multimodal fusion features and the target, ensuring the stable enhancement of similarity for matching instances while lowering the similarity for non-matching instances.
  - 🏆 **State-of-the-Art Performance**: Achieves optimal performance on all metrics across two CIR benchmark datasets, FashionIQ and CIRR, demonstrating strong cross-domain generalization ability.

[⬆ Back to top](#top)

## 🏗️ Architecture

<p align="center">
  <img src="[TODO: assets/HINT-framework.png]" alt="HINT architecture" width="1000">
  <figcaption><strong>Figure 1.</strong> HINT framework consists of three modules: (a) Dual Context Extraction, (b) Quantification of Contextual Relevance, (c) Dual-Path Consistency Constraints. </figcaption>
</p>

[⬆ Back to top](#top)

## 🏃‍♂️ Experiment-Results

### CIR Task Performance

#### FashionIQ:

<caption><strong>Table 1.</strong> Performance comparison on FashionIQ dataset. HINT achieves a notable relative increase of approximately 9.74% in the average R@10.</caption>

<p align="center">
  <img src="[TODO: ./assets/results-fiq.png]" alt="FashionIQ Results">
</p>

#### CIRR:
<caption><strong>Table 2.</strong> Performance comparison on the CIRR test set. HINT achieves a 1.74% improvement in R@1 on CIRR dataset.</caption>

<p align="center">
  <img src="[TODO: ./assets/results-cirr.png]" alt="CIRR Results">
</p>

[⬆ Back to top](#top)

---

## Table of Contents

- [Introduction](#-introduction)
- [News](#-news)
- [Key Features](#-key-features)
- [Architecture](#️-architecture)
- [Experiment Results](#️-experiment-results)
- [Install](#-install)
- [Data Preparation](#-data-preparation)
- [Quick Start](#-quick-start)
  - [Training](#1-training)
  - [Testing](#2-testing)
- [Acknowledgement](#-acknowledgement)
- [Contact](#️-contact)
- [Citation](#️-citation)

---

## 📦 Install

**1. Clone the repository**

```bash
git clone https://github.com/[TODO: 你的GitHub用户名]/HINT.git
cd HINT
```

**2. Setup Python Environment**

The code is evaluated on **Python 3.8** and **CUDA 12.1** (matching our base framework requirements). We recommend using Anaconda:

```bash
conda create -n hint python=3.8
conda activate hint

# Install PyTorch
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies (HINT relies on BLIP-2 architectures)
pip install open-clip-torch==2.24.0 scikit-learn==1.3.2 transformers==4.25.0 salesforce-lavis==1.0.2 timm==0.9.16
```

[⬆ Back to top](#top)

-----

## 📂 Data Preparation

We evaluated our framework on two standard datasets: [FashionIQ](https://github.com/XiaoxiaoGuo/fashion-iq) and [CIRR](https://github.com/Cuberick-Orion/CIRR). Please download the datasets first.

*(Note: Data structure instructions are identical to standard CIR pipelines. Please refer to the official dataset repositories for exact directory formatting).*

[⬆ Back to top](#top)

-----

## 🚀 Quick Start

### 1. Training

Our model is trained using the AdamW optimizer. The hyper-parameter $\lambda$ for the loss function is set to 0.2.

**Training on FashionIQ:**

```bash
python train.py \
    --dataset fashioniq \
    --fashioniq_path "[TODO: /path/to/FashionIQ/]" \
    --model_dir "./checkpoints/fashioniq_hint" \
    --batch_size [TODO: 填写Batch Size] \
    --num_epochs [TODO: 填写Epoch数量] \
    --lr 2e-5
```

**Training on CIRR:**

```bash
python train.py \
    --dataset cirr \
    --cirr_path "[TODO: /path/to/CIRR/]" \
    --model_dir "./checkpoints/cirr_hint" \
    --batch_size [TODO: 填写Batch Size] \
    --num_epochs [TODO: 填写Epoch数量] \
    --lr 2e-5
```

> **💡 Tips:**
> - Our model extracts features using the pre-trained BLIP-2. We perform training on a single NVIDIA V100 GPU with 32GB of memory.

### 2. Testing

To generate the prediction files on the CIRR dataset for submission to the CIRR Evaluation Server, run the testing script:

```bash
python src/cirr_test_submission.py checkpoints/cirr_hint/
```

[⬆ Back to top](#top)

-----

## 🤝 Acknowledgement

The implementation of this project utilizes the pre-trained vision-language features from BLIP-2 and references the [LAVIS](https://github.com/salesforce/LAVIS) framework. We express our sincere gratitude to these open-source contributions!

[⬆ Back to top](#top)

## ✉️ Contact

For any questions, issues, or feedback, please open an [issue](https://github.com/[TODO: 你的GitHub用户名]/HINT/issues) on GitHub or reach out to us at `[TODO: 你的邮箱地址]`.

[⬆ Back to top](#top)

## 🔗 Related Projects

*Ecosystem & Other Works from our Team*
<table style="width:100%; border:none; text-align:center; background-color:transparent;">
  <tr style="border:none;">
    <td style="width:30%; border:none; vertical-align:top; padding-top:30px;">
      <img src="[TODO: /assets/logos/retrack-logo.png]" alt="ReTrack" style="height:65px; width:auto; border-radius:8px; margin-bottom:8px;"><br>
      <b>ReTrack (AAAI'26)</b><br>
      <span style="font-size: 0.9em;">
        <a href="[https://lee-zixu.github.io/ReTrack.github.io/](https://lee-zixu.github.io/ReTrack.github.io/)" target="_blank">Web</a> | 
        <a href="[https://github.com/Lee-zixu/ReTrack](https://github.com/Lee-zixu/ReTrack)" target="_blank">Code</a> | 
        <a href="[https://ojs.aaai.org/index.php/AAAI/article/view/39507](https://ojs.aaai.org/index.php/AAAI/article/view/39507)" target="_blank">Paper</a>
      </span>
    </td>
    <td style="width:30%; border:none; vertical-align:top; padding-top:30px;">
      <img src="[TODO: /assets/logos/intent-logo.png]" alt="INTENT" style="height:65px; width:auto; border-radius:8px; margin-bottom:8px;"><br>
      <b>INTENT (AAAI'26)</b><br>
      <span style="font-size: 0.9em;">
        <a href="[https://zivchen-ty.github.io/INTENT.github.io/](https://zivchen-ty.github.io/INTENT.github.io/)" target="_blank">Web</a> | 
        <a href="[https://github.com/ZivChen-Ty/INTENT](https://github.com/ZivChen-Ty/INTENT)" target="_blank">Code</a> | 
        <a href="[https://ojs.aaai.org/index.php/AAAI/article/view/39181](https://ojs.aaai.org/index.php/AAAI/article/view/39181)" target="_blank">Paper</a>
      </span>
    </td>  
    <td style="width:30%; border:none; vertical-align:top; padding-top:30px;">
      <img src="[TODO: /assets/logos/encoder-logo.png]" alt="ENCODER" style="height:65px; width:auto; border-radius:8px; margin-bottom:8px;"><br>
      <b>ENCODER (AAAI'25)</b><br>
      <span style="font-size: 0.9em;">
        <a href="[https://sdu-l.github.io/ENCODER.github.io/](https://sdu-l.github.io/ENCODER.github.io/)" target="_blank">Web</a> | 
        <a href="[https://github.com/Lee-zixu/ENCODER](https://github.com/Lee-zixu/ENCODER)" target="_blank">Code</a> | 
        <a href="[https://ojs.aaai.org/index.php/AAAI/article/view/32541](https://ojs.aaai.org/index.php/AAAI/article/view/32541)" target="_blank">Paper</a>
      </span>
    </td>
  </tr>
</table>

## 📝⭐️ Citation

If you find our work or this code useful in your research, please consider leaving a **Star**⭐️ or **Citing**📝 our paper 🥰. Your support is our greatest motivation!

```bibtex
@inproceedings{HINT2026,
  title={HINT: COMPOSED IMAGE RETRIEVAL WITH DUAL-PATH COMPOSITIONAL CONTEXTUALIZED NETWORK},
  author={Zhang, Mingyu and Li, Zixu and Chen, Zhiwei and Fu, Zhiheng and Zhu, Xiaowei and Nie, Jiajia and Wei, Yinwei and Hu, Yupeng},
  booktitle={Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2026}
}
```

[⬆ Back to top](#top)

<div align="center">
  <br><br>

  <a href="[https://github.com/](https://github.com/)[TODO: 你的GitHub用户名]/HINT">
    <img src="[https://img.shields.io/badge/](https://img.shields.io/badge/)⭐_Star_US-000000?style=for-the-badge&logo=github&logoColor=00D9FF" alt="Star">
  </a>
  <a href="[https://github.com/](https://github.com/)[TODO: 你的GitHub用户名]/HINT/issues">
    <img src="[https://img.shields.io/badge/](https://img.shields.io/badge/)🐛_Report_Issues-000000?style=for-the-badge&logo=github&logoColor=FF6B6B" alt="Issues">
  </a>

  <br><br>
<a href="[https://github.com/](https://github.com/)[TODO: 你的GitHub用户名]/HINT">
    <img src="[https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=22&pause=1000&color=00D9FF&center=true&vCenter=true&width=500&lines=Thank+you+for+visiting+HINT!;Looking+forward+to+your+attention](https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=22&pause=1000&color=00D9FF&center=true&vCenter=true&width=500&lines=Thank+you+for+visiting+HINT!;Looking+forward+to+your+attention)!" alt="Typing SVG">
  </a>
</div>
