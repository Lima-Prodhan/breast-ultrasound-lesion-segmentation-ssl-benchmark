# Semantic Segmentation Benchmarking and Self-Supervised Label-Efficiency Study for Breast Ultrasound Lesion Segmentation

This project benchmarks supervised semantic segmentation models for breast ultrasound lesion segmentation and then studies whether self-supervised pretraining can recover a substantial portion of the fully supervised performance when fewer labelled images are available.

## Project at a glance

The study uses the **BCSD-2024 Breast Ultrasound Lesion Segmentation** dataset with **1,521 image-mask pairs from 10 patients**. The final split is patient-level (seed 42) with **1,035 train**, **318 validation**, and **168 test** images.

The project is a two-part study.
Part A establishes a supervised benchmark among:
- DeepLabV3-ResNet50
- SegFormer-B0
- YOLOv26n-Sem

SegFormer-B0 is the strongest Part-A model on the test set reaching **mIoU 0.8758**, **lesion IoU 0.7579** and **mean Dice 0.9296**.

Part B reuses the same split but changes the roles of the partitions:
- **1,035 train images:** unlabelled SSL pretraining corpus
- **318 validation images:** labelled downstream fine-tuning set
- **168 test images:** monitoring + final evaluation

Four self-supervised methods are compared with the same SegFormer-style downstream decoder design:
- SimCLR
- BYOL
- MAE
- DINOv2

DINOv2 is the best-performing SSL method in the comparison with **mIoU 0.7633**, corresponding to **87.2% of the Part-A SegFormer-B0 mIoU** while using **30.7% of the Part-A labelled training set**.

## Results

|        Method       | Test mIoU  | Lesion IoU |  Mean Dice | Labelled images | mIoU recovered vs. Part-A |
|---------------------|-----------:|-----------:|-----------:|----------------:|--------------------------:|
| Part-A SegFormer-B0 | **0.8758** | **0.7579** | **0.9296** |      1,035      |          100.0%           |
|       DINOv2        | **0.7633** |   0.5408   |   0.8474   |      318        |          87.2%            |
|       SimCLR        | **0.7232** |   0.4602   |   0.8117   |      318        |          82.6%            |
|       BYOL          | **0.7102** |   0.4352   |   0.7995   |      318        |          81.1%            |
|       MAE           | **0.4900** |   0.0001   |   0.4950   |      318        |          55.9%            |

DINOv2 recovered **87.2%** of the full-supervision SegFormer-B0 mIoU while using **318 labelled images** equal to **30.7%** of the 1,035-image Part-A training set. The other comparison part of the result is in `final comparison results`

## Methodology

### Part A - supervised benchmark

All supervised models use 512x512 inputs. The training pipeline uses synchronized image-mask augmentation with Albumentations; validation and test use deterministic resize only.

**SegFormer-B0** starts from `nvidia/segformer-b0-finetuned-ade-512-512`, replaces the segmentation classifier for the 2-class task and trains with AdamW plus weighted cross-entropy and Dice loss.

**DeepLabV3-ResNet50** uses a COCO-pretrained backbone with a custom 2-class head and an auxiliary loss branch.
**YOLOv26n-Sem** uses the Ultralytics semantic segmentation implementation with a Cityscapes-pretrained checkpoint.

### Part B - label-efficiency study

The 1,035-image Part-A training set is used without masks for SSL pretraining. Only the 318-image validation split is labelled for downstream fine-tuning. The SSL encoders are connected to a SegFormer-style All-MLP decode head through 1x1 convolutional adapters where required.

The four SSL experiments use:
- **SimCLR:** ResNet-50, NT-Xent objective, 50 epochs.
- **BYOL:** ResNet-50 with separate online and target networks; the target network is frozen and updated from the online network using EMA, 50 epochs.
- **MAE:** ViT-Base/16 using `facebook/vit-mae-base`, masked-patch reconstruction, 50 epochs.
- **DINOv2:** ViT-S/14 using `facebook/dinov2-small`, student-teacher self-distillation with 2 global and 4 local crops, 50 epochs.

All the Part-B downstream models use fine-tuned encoders (`frozen_encoder=False`) and a weighted CE + Dice segmentation objective.

## Repository layout

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 00_eda_and_data_prep.ipynb
│   ├── 01_deeplabv3_resnet50.ipynb
│   ├── 02_segformer_b0.ipynb
│   ├── 03_yolov26n_sem_and_part_a_comparison.ipynb
│   ├── 04_part_b_data_prep.ipynb
│   ├── 05_simclr.ipynb
│   ├── 06_byol.ipynb
│   ├── 07_mae.ipynb
│   ├── 08_dinov2.ipynb
│   └── 09_final_comparison.ipynb
├── final comparison results/
│   └── final_comparison.csv
├── report/
│   └──_Project_Report.pdf
└── docs/
    ├── KAGGLE_NOTEBOOKS_URLs.md
    ├── NOTEBOOK_RUN_ORDER.md

```

## Dataset

The BCSD-2024 dataset comprises 10 patients of mammography. The dataset's performance is assessed using the proposed spatial attention mechanism (SAM) for breast lesion segmentation.

**Dataset Link:** https://data.mendeley.com/datasets 

The repository does not redistribute **the BCSD-2024 dataset**. The notebooks expect the dataset through their Kaggle environment paths.
The project report describes the masks as binary after grayscale thresholding at 127. The EDA reports severe pixel imbalance in the sampled masks: approximately **99% background pixels vs. 1% lesion pixels**.

# Limitations

## Evaluation-design caveat

Part B uses the test split for both monitoring/checkpoint selection and final evaluation. The project report explicitly notes that this can make the final reported metrics somewhat optimistic relative to an untouched test set.

## Dataset limitations

The project uses only 10 patients. The EDA reports 1,521 image-mask pairs and a highly imbalanced sampled pixel distribution of roughly 99% background and 1% lesion. This makes lesion IoU and mIoU more informative than raw pixel accuracy for the main segmentation target.

The EDA also found non-uniform sampled image dimensions and two image-mask shape mismatches; the loaders resize mismatched masks with nearest-neighbour interpolation.

## Kaggle notebooks

All 10 notebook links of the final project are in `docs/KAGGLE_NOTEBOOKS_URLs.md`

**Team Members:**
- Lima Prodhan        (2021-3-60-180)
- Mahbub Hasan        (2021-3-60-261)
- Sumaiya Hoque Riyan (2022-1-60-111)

**Course: Digital Image Processing, East West University.**
