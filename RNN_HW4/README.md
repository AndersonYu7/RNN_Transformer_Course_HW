# RNN_HW4

## Project Summary

This project focuses on image classification using the CIFAR-10 dataset and aims to compare the performance of two transformer-based architectures: Vision Transformer (ViT) and SWIN Transformer. The goal is to evaluate how these models perform on a standard image recognition task and to visualize their decision-making processes using Grad-CAM.

I implemented and compared the following models:

* ViT (Vision Transformer)
* SWIN Transformer

The CIFAR-10 dataset includes 10 classes of 32x32 color images. To be compatible with the input requirements of ViT and SWIN, all images were resized to 224x224 and normalized.
The project includes dataset preparation, model fine-tuning, evaluation, Grad-CAM visualization, and a comparative analysis.

## Training and Testing
To train and evaluate the models, use the following notebook:

### ViT Models
- [`ViT_SWIN_ViT.ipynb`](ViT_SWIN_ViT.ipynb): ViT model trained with learning rate = **5e-5**.
- [`ViT_SWIN_ViT_lr7e-4.ipynb`](ViT_SWIN_ViT_lr7e-4.ipynb): ViT model trained with learning rate = **7e-4**.

### SWIN Transformer Models
- [`ViT_SWIN_SWIN.ipynb`](ViT_SWIN_SWIN.ipynb): SWIN Transformer model trained with learning rate = **5e-5**.
- [`ViT_SWIN_SWIN_lr7e-4.ipynb`](ViT_SWIN_SWIN_lr7e-4.ipynb): SWIN Transformer model trained with learning rate = **7e-4**.

This notebook contains code for:
* Data loading and preprocessing
* Model initialization and training
* Evaluation (accuracy, error rate, confusion matrix)
* Grad-CAM visualization

Follow the step-by-step instructions in the notebook to reproduce the results.

## Results

A detailed analysis is available in the [RNN_HW4_Report.pdf](RNN_HW4_Report.pdf) document, including training curves, Grad-CAM visualizations, and model comparisons.

The following table summarizes the test set accuracy results under different learning rate settings:

| Model            | Learning Rate | Accuracy (%) |
| ---------------- | ------------- | ------------ |
| ViT              | 5e-5          | 98.78        |
| ViT              | 7e-4          | 83.84        |
| SWIN Transformer | 5e-5          | **98.93**    |
| SWIN Transformer | 7e-4          | 97.22        |



