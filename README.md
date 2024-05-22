# Integrating Neural-Symbolic Reasoning with Variational Causal Inference Network for Explanatory Visual Question Answering

[Dizhan Xue](https://scholar.google.com/citations?user=V5Aeh_oAAAAJ), [Shengsheng Qian](https://scholar.google.com/citations?user=bPX5POgAAAAJ), and [Changsheng Xu](https://scholar.google.com/citations?user=hI9NRDkAAAAJ).

**MAIS, Institute of Automation, Chinese Academy of Sciences**

![](https://img.shields.io/badge/Status-building-brightgreen)
![GitHub stars](https://img.shields.io/github/stars/LivXue/VCIN?color=yellow&label=Stars)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FLivXue%2FVCIN&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Hits&edge_flat=false)](https://hits.seeyoufarm.com)


### Data
1. Download the [GQA Dataset](https://cs.stanford.edu/people/dorarad/gqa/download.html).
2. Download the [GQA-OOD Dataset](https://github.com/gqa-ood/GQA-OOD)
3. Download the [bottom-up features](https://github.com/airsplay/lxmert) and unzip it.
4. Extracting features from the raw tsv files (**Important**: You need to run the code in Linux):
  ```
  python ./preprocessing/extract_tsv.py --input $TSV_FILE --output $FEATURE_DIR
  ```
5. We provide the annotations of GQA-REX Dataset in `model/processed_data/converted_explanation_train_balanced.json` and `model/processed_data/converted_explanation_val_balanced.json`.
6. (Optional) You can construct the GQA-REX Dataset by yourself following [instructions by its authors](https://github.com/szzexpoi/rex).
7. Download our [generated programs]() of the GQA dataset.
8. (Optional) You can generate programs by yourself following [this project](https://github.com/wenhuchen/Meta-Module-Network).

### Models
We provide four models in `model/model/model.py`.

#### Two baselines:
1. REX-VisualBert is from [this project](https://github.com/szzexpoi/rex).
2. REX-LXMERT replaces the backbone VisualBert of REX-VisualBert by LXMERT.

#### Two our models (using LXMERT as backbone):
1. VCIN is proposed in our ICCV 2023 paper "Variational Causal Inference Network for Explanatory Visual Question Answering".
2. Pro-VCIN is proposed in TPAMI 2024 paper "Integrating Neural-Symbolic Reasoning with Variational Causal Inference Network for Explanatory Visual Question Answering".

### Training and Test
Before training, you need to first generate the dictionary for questions, answers, explanations, and program modules:
  ```
  cd ./model
  python generate_dictionary --question $GQA_ROOT/question --exp $EXP_DIR  --pro $PRO_DIR --save ./processed_data
  ```

The training process can be called as:
  ```
  python main.py --mode train --anno_dir $GQA_ROOT/question --ood_dir $OOD_ROOT/data --sg_dir $GQA_ROOT/scene_graph --lang_dir ./processed_data --img_dir $FEATURE_DIR/features --bbox_dir $FEATURE_DIR/box --checkpoint_dir $CHECKPOINT --explainable True
  ```
To evaluate on the GQA-testdev set or generating submission file for online evaluation on the test-standard set, call:
  ```
  python main.py --mode $MODE --anno_dir $GQA_ROOT/question --ood_dir $OOD_ROOT/data --lang_dir ./processed_data --img_dir $FEATURE_DIR/features --weights $CHECKPOINT/model_best.pth --explainable True
  ```
and set `$MODE` to `eval` or `submission` accordingly.

### Reference
If you find our papers or code helpful, please cite it as below. Thanks!
```
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
