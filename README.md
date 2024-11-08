# Automatic Segmentation of Head & Neck Tumors for MRI-Guided Adaptive Radiotherapy

Welcome to the repository for our solution to the **HNTS-MRG24 MICCAI Challenge**, where we achieved first place in Task 1 (pre-radiotherapy segmentation) using advanced deep learning methods for automatic segmentation of primary gross tumor volume (GTVp) and metastatic node gross tumor volume (GTVn) in MRI scans. This repository provides a step-by-step guide to reproduce our results, including model training, ensembling, evaluation, and visualization.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Workflow](#workflow)
  - [Data Preparation](#data-preparation)
  - [Model Training](#model-training)
  - [Ensemble Strategy](#ensemble-strategy)
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

## Installation

To set up the environment and install dependencies:
```bash
# Clone the repository
git clone https://github.com/yourusername/HNTS-MRG24-Segmentation
cd HNTS-MRG24-Segmentation

# Install dependencies
pip install -r requirements.txt
