
<div align="center">

# Integrating Neural-Symbolic Reasoning with Variational Causal Inference Network for Explanatory Visual Question Answering

[![Status](https://img.shields.io/badge/Status-maintained-brightgreen.svg)](https://github.com/LivXue/VCIN)
[![GitHub stars](https://img.shields.io/github/stars/LivXue/VCIN?color=yellow&amp;label=Stars)](https://github.com/LivXue/VCIN/stargazers)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Paper ICCV 2023](https://img.shields.io/badge/Paper-ICCV%202023-red)](https://openaccess.thecvf.com)
[![Paper TPAMI 2024](https://img.shields.io/badge/Paper-TPAMI%202024-orange)](https://ieeexplore.ieee.org)

</div>

---

## 📋 Table of Contents

- [About](#-about)
- [News](#-news)
- [Installation](#-installation)
- [Data Preparation](#-data-preparation)
- [Models](#-models)
- [Training &amp; Evaluation](#-training--evaluation)
- [Results](#-results)
- [Citation](#-citation)
- [Contact](#-contact)

---

## 🔍 About

This repository contains the official implementation of two papers:

1. **VCIN** (ICCV 2023): *Variational Causal Inference Network for Explanatory Visual Question Answering*
2. **Pro-VCIN** (TPAMI 2024): *Integrating Neural-Symbolic Reasoning with Variational Causal Inference Network for Explanatory Visual Question Answering*

### Authors

[Dizhan Xue](https://scholar.google.com/citations?user=V5Aeh_oAAAAJ), 
[Shengsheng Qian](https://scholar.google.com/citations?user=bPX5POgAAAAJ), 
and [Changsheng Xu](https://scholar.google.com/citations?user=hI9NRDkAAAAJ)

### Affiliation

**State Key Laboratory of Multimodal Artificial Intelligence Systems, Institute of Automation, Chinese Academy of Sciences**

---

## 📰 News

- **2024**: Pro-VCIN accepted to TPAMI 2024
- **2023**: VCIN accepted to ICCV 2023

---

## 💻 Installation

Clone this repository and set up the environment:

```bash
git clone https://github.com/LivXue/VCIN.git
cd VCIN

# Create conda environment
conda env create -f environment.yaml
conda activate vcin
```

---

## 📦 Data Preparation

Follow these steps to prepare the datasets:

### 1. Download Datasets

- **GQA Dataset**: [Download here](https://cs.stanford.edu/people/dorarad/gqa/download.html)
- **GQA-OOD Dataset**: [Download here](https://github.com/gqa-ood/GQA-OOD)

### 2. Download Features

Download the [bottom-up features](https://github.com/airsplay/lxmert) and unzip them.

### 3. Extract Features

**Important**: You need to run this in Linux:

```bash
python ./preprocessing/extract_tsv.py --input $TSV_FILE --output $FEATURE_DIR
```

### 4. GQA-REX Annotations

We provide the annotations of GQA-REX Dataset in:
- `model/processed_data/converted_explanation_train_balanced.json`
- `model/processed_data/converted_explanation_val_balanced.json`

*(Optional)* You can construct the GQA-REX Dataset by yourself following [instructions by its authors](https://github.com/szzexpoi/rex).

### 5. Generated Programs

Download our generated programs of the GQA dataset from [Google Drive](https://drive.google.com/drive/folders/1irW8aVOBm0CmOxN6ovVBYlTTQvqn1NLx?usp=sharing).

*(Optional)* You can generate programs by yourself following [this project](https://github.com/wenhuchen/Meta-Module-Network).

---

## 🤖 Models

We provide four models in `model/model/model.py`:

### Baselines

| Model | Description | Backbone |
|-------|-------------|----------|
| REX-VisualBert | From [REX project](https://github.com/szzexpoi/rex) | VisualBert |
| REX-LXMERT | REX-VisualBert with LXMERT backbone | LXMERT |

### Our Methods

| Model | Paper | Backbone |
|-------|-------|----------|
| VCIN | ICCV 2023 | LXMERT |
| Pro-VCIN | TPAMI 2024 | LXMERT |

---

## 🚀 Training &amp; Evaluation

### Step 1: Generate Dictionary

Before training, generate the dictionary for questions, answers, explanations, and program modules:

```bash
cd ./model
python generate_dictionary.py --question $GQA_ROOT/question --exp $EXP_DIR --pro $PRO_DIR --save ./processed_data
```

### Step 2: Training

```bash
python main.py --mode train \
    --anno_dir $GQA_ROOT/question \
    --ood_dir $OOD_ROOT/data \
    --sg_dir $GQA_ROOT/scene_graph \
    --lang_dir ./processed_data \
    --img_dir $FEATURE_DIR/features \
    --bbox_dir $FEATURE_DIR/box \
    --checkpoint_dir $CHECKPOINT \
    --explainable True
```

### Step 3: Evaluation

To evaluate on GQA-testdev set or generate submission file:

```bash
python main.py --mode $MODE \
    --anno_dir $GQA_ROOT/question \
    --ood_dir $OOD_ROOT/data \
    --lang_dir ./processed_data \
    --img_dir $FEATURE_DIR/features \
    --weights $CHECKPOINT/model_best.pth \
    --explainable True
```

Set `$MODE` to `eval` or `submission` accordingly.

---

## 📊 Results

*(Add your results table or figures here)*

---

## 📝 Citation

If you find our papers or code helpful, please cite:

```bibtex
@inproceedings{xue2023variational,
  title={Variational Causal Inference Network for Explanatory Visual Question Answering},
  author={Xue, Dizhan and Qian, Shengsheng and Xu, Changsheng},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={2515--2525},
  year={2023}
}

@article{xue2024integrating,
  title={Integrating Neural-Symbolic Reasoning With Variational Causal Inference Network for Explanatory Visual Question Answering},
  author={Xue, Dizhan and Qian, Shengsheng and Xu, Changsheng},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2024},
  publisher={IEEE}
}
```

---

## 📬 Contact

For questions, please open an issue or contact:

- Dizhan Xue: [xuedizhan17@mails.ucas.ac.cn](mailto:xuedizhan17@mails.ucas.ac.cn)

---

<div align="center">

Made with ❤️ by the VCIN Team

</div>

