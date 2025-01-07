Added ipynb of Most Files  :

# CIFAR100 & FMNIST Experiments Repository

This repository contains my experiments on **image classification** using **Autoencoders**, **Siamese Networks**, and **Transfer Learning**. The experiments were conducted on two datasets: **CIFAR-100** and **Fashion MNIST (FMNIST)**. The repository is structured to include different approaches, scores, and supporting documentation for better navigation and understanding.

---

## Table of Contents
- **[Introduction](#introduction)**
- **[Datasets](#datasets)**
- **[Repository Structure](#repository-structure)**
- **[Approaches and Experiments](#approaches-and-experiments)**
  - [CIFAR-100](#cifar-100)
  - [FMNIST](#fmnist)
- **[Scores](#scores)**
- **[References](#references)**
- **[Future Work](#future-work)**

---

## Introduction

This repository is focused on experimenting with **representation learning** and **image similarity learning** for diverse datasets. The key methodologies include:

- **Autoencoders** for dimensionality reduction and feature extraction.
- **Siamese Networks** for learning image similarity.
- **Transfer Learning** with pre-trained models like **ResNet50**.

These approaches are tailored to tackle challenges posed by datasets with varying complexity, such as CIFAR-100's 100-class structure.

---

## Datasets

1. **CIFAR-100**: A dataset containing 60,000 color images across 100 classes, including animals, nature, and objects. Each image has a resolution of 32x32 pixels.
2. **FMNIST (Fashion MNIST)**: A simpler dataset with 10 grayscale image classes, often used for benchmarking deep learning models in fashion-based categories.

---

## Repository Structure

```plaintext
FMNIST/
│
├── Autoencoders/
│   ├── Querying_5_test_Images_Code.py
│   ├── autoencoder_best_model.keras
│   ├── autoencoder_final_model.h5
│   ├── t3_tf_fmnist.py
│   └── testing_9_adv_best_on_full.py
│
├── Siamese_Networks/
│   ├── Learning Tries/
│   │   ├── Siamese_try1.ipynb
│       └── Siamese_try_final_FMNIST.ipynb
│── Transfer Learning MobileNetV2(For FMNIST) 
      └── Transfer Learning MobileNetV2.ipynb

CIFAR100/
│
├── Autoencoders/
│   └── Prac_Attempts/
│       ├── CIFAR100_Autoenc_t3.ipynb
│       └── CIFAR_using_Autoencoder_opt.ipynb
│
├── Siamese/
│   ├── Siamese_CIFAR100_t2.ipynb
│   ├── Transfer Learning (ResNet50)/
│       └── Siamese_with_ResNet50_CIFAR100.ipynb
├── Transfer Learning(ResNet50) with Siamese Networks  
    └── CIFAR_PreTrained_Models (ResNet50).ipynb

Scores/
│   ├── (Screenshots of All approaches model scores)
│
README.md
