---
license: apache-2.0
library_name: diffusers
tags:
- clinical
- oncology imaging
- diffusers
- stable diffusion
base_model:
- stable-diffusion-v2
metrics:
- CLIP scoring
thumbnail: "https://raw.githubusercontent.com/fangdai-dear/Tiger-Model/0fb783bf17ea4770e49f19f24ddd62ac973deff2/dataset/Figure1.png"
pipeline_tag: text-to-image
arxiv: 
---

# Tiger Model
[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)
## Enhancing Diagnostic Generalization of AI Models in Rare Thyroid Cancers: A Clinical Knowledge-Guided Data Augmentation Approach Using Generative Models
__Fang Dai†, Siqiong Yao†\*, Min Wang, Yicheng Zhu, Xiangjun Qiu, Peng Sun, Jisheng Yin, Guangtai Shen, Jingjing Sun, Maofeng Wang, Yun Wang, Zheyu Yang, Jianfeng Sang, Xiaolei Wang, Fenyong Sun\*, Wei Cai\*, Xingcai Zhang\*, Hui Lu\*__\
\* To whom correspondence should be addressed.  
†These authors contributed equally to this work.\
Email: huilu@sjtu.edu.cn.

### Abstract
Artificial intelligence (AI) in oncology imaging struggles with diagnosing rare tumors. Our study identified performance gaps in detecting rare thyroid cancer subtypes using ultrasound, leading to misdiagnoses and adverse prognostic outcomes. Sample scarcity for rare conditions impedes effective model training. Although data augmentation techniques can alleviate sample size constraints, trainable examples cannot encompass the full spectrum of disease manifestations, rendering traditional generative augmentation approaches inadequate. Our approach integrates clinical knowledge with text-image generation, enabling fine-grained control and supplementation of unique features specific to rare subtypes, emphasizing text guidance. This results in augmented samples that more accurately reflect genuine disease cases. Our model, trained on data from 40,571 patients, including 5,099 rare cases, exceeds current state-of-the-art methods, enhancing the AUC for two rare subtypes by 14.64% and 9.45%, respectively. In Turing tests, we achieved 92.2% for authenticity, 90.96% for consistency, and 84.1% for diversity, surpassing competitors by 35.6%. Generalization ability of this methodology was validated on public datasets such as the BrEaST, BUSI, and VinDr-PCXR datasets. This approach mitigates the challenges of data diversity and representativeness for rare diseases, contributing to the model’s generalization ability and diagnostic accuracy, ultimately improving the effectiveness and practical outcomes of medical AI applications.
![figure1](https://raw.githubusercontent.com/fangdai-dear/Tiger-Model/0fb783bf17ea4770e49f19f24ddd62ac973deff2/dataset/Figure1.png)
COPYRIGHT NOTICE: This image is protected by copyright laws and is the property of [Fang Dai/Shanghai Jiao Tong University]. Unauthorized copying, distribution, or use of this image is strictly prohibited. All rights reserved.

#### Research Status: Under Review

## Model architecture
The model architecture is included in the manuscript and will not be displayed before the article is published.
## Install

This project uses requirements.txt.

```sh
$ pip install -r requirements.txt
```
## Datasets
### 1. Thyroid dataset for Tiger Model (The other external validation datasets(BrEaST, BUSI, VinDr-PCXR) are also deployed in folders in the same manner.)

We have shared part of the thyroid ultrasound dataset for verification. Please refer to this article for other studies using this dataset.
If you use this dataset in your research, please cite the following references:
A portion of the data from this article is publicly available on Huggingface ([https://huggingface.co/datasets/FangDai/Thyroid_Ultrasound_Images](https://huggingface.co/datasets/FangDai/Thyroid_Ultrasound_Images). To download this dataset, you must register on Hugging Face and sign our data usage application before gaining access.
![image4](https://huggingface.co/FangDai/Tiger-Model/resolve/main/dataset/image4.png)

Please read the following information for data usage permissions and the conditions for accessing the full dataset.
```
All data that fueled the findings can be found within the article and the Supplementary Information. The Thyroid datasets trained and analyzed during this study are available in a deidentified form to protect patient privacy. The minimum Thyroid dataset required to interpret, verify, and extend the findings of this study has been deposited in Huggingface under accession code https://huggingface.co/datasets/FangDai/Thyroid_Ultrasound_Images. This includes:
- Pre-processed imaging data (ultrasound images with anonymized metadata).
- Clinical feature tables (age, gender, tumor size) with all direct identifiers removed.
Due to ethical restrictions and patient confidentiality agreements, the full dataset (e.g., raw imaging data, detailed clinical records) cannot be made publicly available. This pertains to detailed clinical records and high-resolution imaging data that, even after de-identification, may pose a risk of re-identification given the unique characteristics of thyroid cancer cases. Researchers who wish to access additional data for non-commercial academic purposes may submit a formal request to the corresponding author. Requests will be reviewed by the institutional ethics committee and data custodians. The following conditions apply:
- Purpose: Data will only be shared for research purposes that align with the original study objectives. 
- Access Restrictions: Requesters must sign a data use agreement prohibiting re-identification or redistribution.
- Data Retention: Approved data will be available for 2 years from the date of publication.
```
This dataset contains 900 thyroid ultrasound images, categorized into three subtypes of thyroid carcinoma:
- PTC (Papillary Thyroid Carcinoma) 
- FTC (Follicular Thyroid Carcinoma) 
- MTC (Medullary Thyroid Carcinoma) 
##### The dataset is curated to support medical image classification and segmentation tasks, particularly for deep learning applications in thyroid cancer diagnosis.
It is curated to support medical image classification, particularly for AI applications in thyroid cancer diagnosis.
#### Citation
```bibtex
@article{yao2024enhancing,
  title={Enhancing the fairness of AI prediction models by Quasi-Pareto improvement among heterogeneous thyroid nodule population},
  author={Yao, Siqiong and Dai, Fang and Sun, Peng and Zhang, Weituo and Qian, Biyun and Lu, Hui},
  journal={Nature Communications},
  volume={15},
  number={1},
  pages={1958},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
```
```sh
├─dataset
    └─training data
        ├─init_image
            20191101_094744_1.png
            ... ...
            metadata.jsonl
        ├─condition_FG
            20191101_094744_1.png
            ... ...
        ├─condition_BG
            20191101_094744_1.png
            ... ...
```
### 2. Thyroid dataset for Resnet Model (The other external validation datasets(BrEaST, BUSI, VinDr-PCXR) are also deployed in folders in the same manner.)
```sh
├─dataset
│  ├─Renset training data
│  │  └─PTC
│  │      └─train
│  │          └─0
│  │              figure1.jpg
                  ... ...
│  │          └─1
│  │              figure2.jpg
                  ... ...
│  │      └─valid
│  │          └─0
│  │              figure3.jpg
                  ... ...
│  │          └─1
│  │              figure4.jpg
                  ... ...
│  │      └─test
│  │          └─0
│  │              figure5.jpg
                  ... ...
│  │          └─1
│  │              figure6.jpg
                  ... ...
│  │  └─FTC
        ... ...
│  │  └─MTC
        ... ...
```
Partial thyroid ultrasonography data used in this study are subject to privacy restrictions, but may be anonymized and made available upon reasonable request to the corresponding author.

## Training data preparation
metadata.josnl: The file is placed under folder dataset/training data/init_image, each of which acts as the file name of the image, the associated condition file, and text information to facilitate the subsequent import of the model
```sh
| {"file_name": "20191101_094744_1.png", "condition_FG": "../training data/condition_nd/20191101_094744_1.png", "condition_BG": "../training data/condition_bg/20191101_094744_1.png", "text_nd": "papillary, wider-than-tall, clear, regular", "text_bg": "145.819221, 51.008308, 2.096069"}\
| {... ...}
```
## Installation
We recommend installing Tiger Model in a virtual environment via Conda. For more detailed information about installing PyTorch, please refer to the official documentation.

### PyTorch-diffusers (including Stable Diffusion, ControlNet, Transformer)
With `pip` (official package):
```bash
pip install --upgrade diffusers[torch]
```
With `conda` (maintained by the community):
```sh
conda install -c conda-forge diffusers
```
### PyTorch-Others
```sh
conda list -e > requirements.txt
```

## Tiger Model Coarse-Training
Coarse-Training: based on the Stable Diffusion (SD) model . Training utilizes ultrasound images and corresponding textual reports (Image + Prompt) as inputs. During this phase, the model is able to generate coarse-grained image features based on text. 
```sh
$ sh ./Tiger-Corase.sh
```

## Tiger Model Fine-Training
To optimize details, utilized the trainable Encoder weights from the Coarse-Training 
model , and employed the conditional control method similar to ControlNet but with some differences.
```sh
$ sh ./Tiger-Fine.sh
```


## Trained Model Release: **Tiger**

This repository contains the trained model for **Tiger** designed for **thyroid image generation**. The model was trained on **stable diffusion** using **PyTorch**.

### Model Details
- **Model Architecture**: [stable-diffusion-v2]
- **Input Size**: [224x224 for images]
- **Output**: [image]
- **Framework**: [PyTorch]
- **Download Link 1**: [google drive link to download the Coarse-Training model](https://drive.google.com/file/d/1i5ZBvR5dxEf4Oe-bN51EyNymWW-0lrCG/view?usp=drive_link)]
- **Download Link 2**: [google drive link to download the Fine-Training model](https://drive.google.com/file/d/14PzYy12BYQ5A_OCAGppO0QDzgJklYlhk/view?usp=drive_link)]

## Tiger Model Inference
Tiger Model's application scenarios (inference) can be divided into two categories (Supplementary Fig.3). The first type is Diversify Inference, which involves generating thyroid feature textual prompts based on prompt input combinations. Tiger Model generates synthetic images based on the prompt content, controlling the synthesis of corresponding fine-grained foreground-background features within the model. The second type is Reference Inference, where the input comprises real images. Tiger Model generates images consistent with the subtype of the input image. Both generation scenarios allow for the control of corresponding foreground-background features as needed during the generation process. 
```sh
$ python Tiger Model/generation.py
```

##  Binary classification Resnet50 training
In the training stage, the generated image and the real image are mixed together to train the classification model.
```sh
$ sh ResnNet_main.sh
```

## Evaluation criteria
### CLIP score
The CLIP scoring criteria involve [training](https://github.com/revantteotia/clip-training) a CLIP model and calculating the CLIP score based on the corresponding CLIP values from the model. For specific calculation methods, please refer to the appendix. The CLIP training code is referenced from this study.

### Moso score
We employ the [MoSo score](https://github.com/hrtan/MoSo) to control the quality of generated images, which measures the change in the optimal empirical risk after the exclude of a particular sample from the training set.

## Reference
All references are listed in the article.

## Licence
The code is distributed under the Apache License 2.0. It can be used for non-commercial purposes only after the publication of the article. For any commercial use, please contact the author for permission.