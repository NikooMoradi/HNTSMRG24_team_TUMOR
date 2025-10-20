# Comparative Analysis of nnUNet and MedNeXt for Head and Neck Tumor Segmentation in MRI-guided Radiotherapy

Welcome to the repository for our solution to the **HNTS-MRG24 MICCAI Challenge**, where we achieved first place in Task 1 (pre-radiotherapy segmentation) using advanced deep learning methods for automatic segmentation of primary gross tumor volume (GTVp) and metastatic node gross tumor volume (GTVn) in MRI scans. This repository provides a step-by-step guide to reproduce our results, including model training, ensembling, evaluation, and visualization.

For more details, please refer to our paper:

- **Title**: _Automatic Segmentation Frameworks for Head & Neck Tumors Using MRI Scans_  
- **Authors**: Nikoo Moradi et al.

If you use this repository or any part of our work, please cite the following publication:

> **Moradi, N. et al. (2025).**  
> *Comparative Analysis of nnUNet and MedNeXt for Head and Neck Tumor Segmentation in MRI-Guided Radiotherapy.*  
> In: Wahid, K.A., Dede, C., Naser, M.A., Fuller, C.D. (eds) *Head and Neck Tumor Segmentation for MR-Guided Applications (HNTSMRG 2024).*  
> Lecture Notes in Computer Science, vol 15273. Springer, Cham.  
> [https://doi.org/10.1007/978-3-031-83274-1_10](https://doi.org/10.1007/978-3-031-83274-1_10)


## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Workflow](#workflow)
  - [Model Training](#model-training)
  - [Evaluation](#evaluation)
- [Results](#results)
- [References](#references)

## Project Overview

The **HNTS-MRG24 MICCAI Challenge** focuses on the segmentation of head and neck tumors in MRI-guided adaptive radiotherapy. Our solution utilizes the **nnUNet** and **MedNeXt** models, combined with multi-level ensembling to improve segmentation accuracy. The final solution achieved a **mean DSCagg of 0.8254** for Task 1 and **0.7005** for Task 2.

## Dataset

The project utilizes the **HNTS-MRG24 dataset** provided by the challenge organizers, consisting of MRI images from 150 head and neck cancer patients. The dataset includes:
- **Pre-RT T2-weighted MRI scans** with segmentation masks for GTVp and GTVn.
- **Mid-RT T2-weighted MRI scans** for Task 2, with pre-registered pre-RT images to facilitate training.

> **Note**: Dataset access requires registration at [HNTS-MRG24 Challenge](https://hntsmrg24.grand-challenge.org).

[//]: < ## Installation >

[//]: < To set up the environment and install dependencies:>
[//]: < ```bash>
[//]: < # Clone the repository>
[//]: < git clone https://github.com/yourusername/HNTS-MRG24-Segmentation>
[//]: < cd HNTS-MRG24-Segmentation>

[//]: < # Install dependencies>
[//]: < pip install -r requirements.txt>

This project requires a GPU for model training. Recommended specifications:

- **6 NVIDIA RTX 6000 GPUs**
- **48 GB VRAM**
- **1024 GB RAM**

## Workflow

### Model Training

We utilized **nnUNet** and **MedNeXt** models with the following configurations:

- **Task 1**: Pre-trained on mid-RT and registered pre-RT images, fine-tuned on original pre-RT images.
- **Task 2**: Multi-channel input with mid-RT images, registered pre-RT images, and registered pre-RT segmentation masks.


### Evaluation

The primary evaluation metric is the **Aggregated Dice Similarity Coefficient (DSCagg)**. We report results for each configuration and use per-sample Dice scores to assess model robustness. 

The evaluation code is available in `Evaluation/DSCagg_calculation.ipynb`.

## Results

**Final Submitted Solutions**

- **Task 1 (Pre-RT Segmentation):**  
  **MedNeXt-Small (3×3×3 kernels)** trained on registered pre-RT + mid-RT images and fine-tuned on original pre-RT data.  
  *Mean DSC<sub>agg</sub> = 0.8254 → 1st place.*

- **Task 2 (Mid-RT Segmentation):**  
  **nnUNet Ensemble (FullRes + Cascade)** using multi-channel input of mid-RT, registered pre-RT, and registered pre-RT masks.  
  *Mean DSC<sub>agg</sub> = 0.7005 → 8th place.*

[//]: < Fig 1. Comparison of predictions between nnUNet, MedNeXt, and ground truth. >

[//]: < Additional figures can be found in the `results/` folder.>


## Contact

For questions or support, please contact:

- **Nikoo Moradi** (Email: [nikoomoradi81@gmail.com](mailto:nikoomoradi81@gmail.com))

Feel free to contribute or open issues if you find any bugs or have suggestions!

