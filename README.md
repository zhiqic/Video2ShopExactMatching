# Video2Shop: Exact Matching Clothes in Videos to Online Shopping Images

![Python badge](https://img.shields.io/static/v1?label=python&message=3.6%7C3.7%7C3.8&color=blue)
![PyTorch badge](https://img.shields.io/static/v1?label=pytorch&message=1.6&color=orange)

This repository presents an official PyTorch implementation of the Video2Shop algorithm as proposed in the paper: [Exact Matching Clothes in Videos to Online Shopping Images](https://openaccess.thecvf.com/content_cvpr_2017/html/Cheng_Video2Shop_Exact_Matching_CVPR_2017_paper.html).

## Overview

The Video2Shop algorithm enables exact matching of clothing items from videos to corresponding online shopping images, helping bridge the gap between video content and e-commerce platforms.

## Installation

To ensure a smooth setup, it is recommended to use a conda environment.

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
python train.py -tph <training_phase: 0 first part - 1 fusion nodes - 2 both> -e <num_epochs> -lr <learning-rate> --dataset <path-to-dataset> --gpus 0
```

- `<training_phase>`: Define which part of the model to train.
  - 0: Train the first part
  - 1: Train fusion nodes
  - 2: Train both
- `<num_epochs>`: Number of epochs for training
- `<learning-rate>`: Learning rate for the optimizer
- `<path-to-dataset>`: Path to the dataset directory

## Evaluation

To evaluate the trained model:

```bash
python evaluation.py -ckpt <path-to-checkpoint>
```

- `<path-to-checkpoint>`: Path to the saved model checkpoint

## Contribution

Feel free to raise issues or pull requests if you find any bugs or improvements in the code.

## Citation

If you find this implementation beneficial and utilize it in your research, please cite using the following BibTeX entry:

```
@inproceedings{cheng2017video2shop,
  title={Video2shop: Exact matching clothes in videos to online shopping images},
  author={Cheng, Zhi-Qi and Wu, Xiao and Liu, Yang and Hua, Xian-Sheng},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={4048--4056},
  year={2017}
}
```
