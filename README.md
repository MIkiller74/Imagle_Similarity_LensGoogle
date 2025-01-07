Added ipynb of Most Files  :

#  FMNIST & CIFAR100 Image Similarity Search using Multiple Approaches 

This repository contains my experiments on **image similarity** using **Autoencoders**, **Siamese Networks**, and **Transfer Learning**. The experiments were conducted on two datasets: **CIFAR-100** and **Fashion MNIST (FMNIST)**. The repository is structured to include different approaches, scores, and supporting documentation for better navigation and understanding.

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
```
### 1st is Query Image and next 5 are retrieved.
![Query 1](https://github.com/MIkiller74/Imagle_Similarity_LensGoogle/blob/main/Scores/Query1.jpg)
![Query 2](https://github.com/MIkiller74/Imagle_Similarity_LensGoogle/blob/main/Scores/Query2.jpg)
![Query CIFAR100](https://github.com/MIkiller74/Imagle_Similarity_LensGoogle/blob/main/Scores/Query%20CIFAR100.png)
![Query 5](https://github.com/MIkiller74/Imagle_Similarity_LensGoogle/blob/main/Scores/AAQuery.png)

## Approaches and Experiments
### **FMNIST**
### **PreTrained MobileNet V2**
- MobileNetV2 Transfer pre-trained model was used for Image Similarity.
- Gave decent score of accuracy 85%.

#### **Autoencoders**
- **Objective**: Dimensionality reduction and latent space visualization for fashion categories.
- **Highlights**:
  - Achieved significant results with the best model stored in **`autoencoder_best_model.keras`**.

#### **Siamese Networks**
- **Objective**: Pairwise comparison for image similarity in fashion categories.
- **Experiments**:
  - Initial models focused on **custom architectures**.

    
### **CIFAR-100**

#### **Autoencoders**
- **Objective**: _Reduce dimensionality_ and extract features from CIFAR-100 images.
- **Challenges**: High class diversity and complexity in feature representation.
- **Experiments**: 
  - Tried multiple architectures by varying **filter sizes** and **layers**.
  - **Results**: Limited performance due to the complexity of the dataset.

#### **Siamese Networks**
- **Objective**: Learn pairwise image similarity for classification.
- Did'nt get good score only with siamese but later with Transfer learning with ResNet50 model gave acceptable score.
 ### **Transfer Learning (ResNet50) with Siamese Approach**
  - **Why**: Leverage pre-trained ResNet50 to extract meaningful features from the dataset.
  - **Result**: Achieved decent accuracy and acceptable results for a highly generalized dataset like CIFAR-100.
### **PreTrained Models ResNet50 and Mobilenet V2**
  - Gave bad scores on pre-trained models
---

## Scores

### Added all the Screenshots of the Scores

## References

- [Autoencoders and Dimensionality Reduction](https://arxiv.org/abs/1512.03385) - _A key paper that describes the use of autoencoders for reducing dimensionality and unsupervised learning._
- [Explainable Image Similarity: Integrating Siamese Networks and Grad-CAM](https://arxiv.org/pdf/2310.07678) 
- [Siamese Networks for One-Shot Learning](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) - _A foundational paper introducing Siamese Networks for tasks requiring one-shot learning._
- [Transfer Learning with ResNet](https://arxiv.org/abs/1512.03385) - _An essential reference on ResNet architecture and its applications in transfer learning._
- [FMNIST Dataset Overview](https://github.com/zalandoresearch/fashion-mnist) - _Detailed information on the FMNIST dataset, including benchmarks and evaluation metrics._


