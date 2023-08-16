# Video2Shop: Exact Matching Clothes in Videos to Online Shopping Images

<div align="left">
    <img src="https://img.shields.io/static/v1?label=python&message=3.6%7C3.7%7C3.8&color=blue" alt="Python badge">
    <img src="https://img.shields.io/static/v1?label=pytorch&message=1.6&color=orange" alt="PyTorch badge">
</div>


## Introduction

In today's digital era, videos have taken center stage in how we consume content. From high-impact ads to influencers setting fashion statements, videos are where we see new trends. But how often have you seen an outfit in a video and wondered where to buy it? Enter the **Video2Shop** algorithm, the backbone of the revolutionary **Pailitao** system.

This repository houses an unofficial PyTorch version of the **Video2Shop** algorithm, an innovation making online shopping effortlessly intuitive. See a stunning outfit in a movie or on the street? **Pailitao** ensures it's just a few clicks away from being yours.

<div align="center">
    <img src="https://github.com/zhiqic/Video2Shop/assets/65300431/257b51d5-cb2d-4dbc-9156-715f67f4444a" alt="Pailitao System" width="80%">
</div>

📹 Dive into the nitty-gritty details in our paper presented at **CVPR 2017**: [Video2Shop: Exact Matching Clothes in Videos to Online Shopping Images](https://openaccess.thecvf.com/content_cvpr_2017/html/Cheng_Video2Shop_Exact_Matching_CVPR_2017_paper.html).

🎥 For a firsthand look at **Video2Shop** in action, don't miss this [YouTube demo](https://www.youtube.com/watch?v=FK-m7YXuf5Y).


### Patent Information:
- **Patent:** [US10671851B2](https://patents.google.com/patent/US10671851B2/en)
- **Country:** United States
- **Assignee:** Alibaba Group Holding Ltd
- **Status:** Active, with validity up to 2038-09-29 

## Overview

The Video2Shop algorithm, as its name suggests, is adept at matching clothing items from videos directly to their online shopping counterparts. Not only does this bridge the ever-expanding realm of video content with e-commerce platforms, but it also addresses the latent consumer need for instant gratification in the digital shopping realm.

<div align="center">
    <img src="https://github.com/zhiqic/Video2Shop/assets/65300431/24dcc82a-87bf-4e45-97b6-cc4146b37ecf" alt="AsymNet Framework">
    <p><i>Figure 1. Framework of AsymNet. After clothing detection and tracking, deep visual features are produced by the Image Feature Network (IFN) and Video Feature Network (VFN). These features then undergo pair-wise matching in the similarity network.</i></p>
</div>

## Installation

### Prerequisites

- Python 3.6 or higher
- PyTorch 1.6 with CUDA support
- torchvision
- tensorboardX
- easydict
- tqdm

### Setup Steps

1. Clone the repository:
   ```bash
   git clone <repository-link>
   ```
2. Navigate to the repository folder:
   ```bash
   cd <repository-folder-name>
   ```
3. Create a conda environment and activate it:
   ```bash
   conda create --name video2shop python=3.7
   conda activate video2shop
   ```
4. Install required packages:
   ```bash
   pip install torch==1.6 torchvision tensorboardX easydict tqdm
   ```

## Training

To initiate training on a single GPU:

```bash
python train.py -tph <training_phase> -e <num_epochs> -lr <learning-rate> --dataset <path-to-dataset> --gpus 0
```

## Evaluation

To evaluate the trained model:

```bash
python evaluation.py -ckpt <path-to-checkpoint>
```

## Acknowledgement
Adhering to the ethos of open research and community contribution, this repository welcomes the academic community's feedback. While this is an unofficial and independent reproduction, its primary use case leans towards academic research. Commercial utilization is discouraged. Props to [kyusbok](https://github.com/kyusbok) for the initial replication endeavors. Kindly raise issues or submit pull requests for any enhancements or identified bugs.

## Citation

```
@inproceedings{cheng2017video2shop,
  title={Video2shop: Exact matching clothes in videos to online shopping images},
  author={Cheng, Zhi-Qi and Wu, Xiao and Liu, Yang and Hua, Xian-Sheng},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={4048--4056},
  year={2017}
}
```
