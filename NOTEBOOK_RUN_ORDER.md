# Notebook run order

The project is a two-part pipeline. The original work was executed in Kaggle and later notebooks consume artifacts produced by earlier notebooks.

| Order | Repository notebook | Role | Main output / dependency |
|---:|---|---|---|
| 0 | `00_eda_and_data_prep.ipynb` | Dataset audit, mask analysis, patient-level split, augmentation | `split.csv`; patient-disjoint train/val/test split |
| 1 | `01_deeplabv3_resnet50.ipynb` | Part-A supervised baseline | DeepLabV3 checkpoint + `results.json` |
| 2 | `02_segformer_b0.ipynb` | Part-A supervised benchmark | SegFormer checkpoint + `results.json` |
| 3 | `03_yolov26n_sem_and_part_a_comparison.ipynb` | Part-A YOLO semantic segmentation + consolidated comparison | YOLO result + Part-A comparison |
| 4 | `04_part_b_data_prep.ipynb` | Reassign Part-A split roles for SSL | Unlabelled train / labelled fine-tune / test-monitoring split artifacts |
| 5 | `05_simclr.ipynb` | SimCLR pretraining + downstream segmentation | `results_B.json` entry for SimCLR |
| 6 | `06_byol.ipynb` | BYOL pretraining + downstream segmentation | `results_B.json` entry for BYOL |
| 7 | `07_mae.ipynb` | MAE pretraining + downstream segmentation | `results_B.json` entry for MAE |
| 8 | `08_dinov2.ipynb` | DINOv2-style pretraining + downstream segmentation | `results_B.json` entry for DINOv2 |
| 9 | `09_final_comparison.ipynb` | Consolidated label-efficiency comparison + error overlap analysis | `final_comparison.csv` + chart |

