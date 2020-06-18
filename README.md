# Dissecting Deep Networks into an Ensemble of GenerativeClassifiers for Robust Predictions
This repository contains the demo code of the paper: **Dissecting Deep Networks into an Ensemble of GenerativeClassifiers for Robust Predictions**. The proposed method called **REGroup**, which stands for *Rank-aggregating Ensemble of Generative Classifiers for Robust Predictions*. REGroup is a **simple**, **scalable** (from CIFAR10 to ImageNet),  and **practical**  defense strategy that is model agnostic and does not require any re-training or fine-tuning. We suggest to use REGroup at test time to make a pre-trained network robust to adversarial perturbations
If you use this repository or REGroup, please please consider citing:.

    @inproceedings{plummerCITE2018,
	Author = {Bryan A. Plummer and Paige Kordas and M. Hadi Kiapour and Shuai Zheng and Robinson Piramuthu and Svetlana Lazebnik},
	Title = {Conditional Image-Text Embedding Networks},
	Booktitle  = {The European Conference on Computer Vision (ECCV)},
	Year = {2018}
    }
# Salient features of REGroup
  - Convert a **pre-trained classifier** into **robust classifier**.
  - REGroup is a test time replacement of Softmax. 
  - Pre-trained model with REGroup testing strategy makes the classfier robust to several adversarial attacks.
  - REGroup is *model agnostics* (VGG, RESNET etc)
  - REGroup is *attack method agnostic* (Gradient Based/Free, White/Black Box, Spatial, Boundary attack etc).
  - Doesn't require any Adversarial training, re-training or finetuning.


# Requirements
  - Pytorch 
  - numpy, scipy 
  - matplotlib 
  - Jupyter notebook 
  - foolbox (version 2.4.0)
  


# Steps to run the demo
- clone the repository.
- Download CIFAR10 data.
- Open jupyter notebook

```sh
$ git clone https://github.com/lokender/REGroup.git
$ wget 
$ jupyter notbeook
```

# To-dos?
  - [**Done**] **Classifier**: VGG19, **Dataset**: CIFAR10  ( *Released* )
  - [**To-do**] **Classifier**: VGG19, **Dataset**: ImageNet ( *Will be released soon* )
  - [**To-do**] **Classifier**: ResNet, **Dataset**: CIFAR10 ( *Will be released soon* )
  - [**To-do**]  **Classifier**: ResNet, **Dataset**: ImageNet ( *Will be released soon* )
  - [**To-do**] **Classifier**: Inception, **Dataset**: ImageNet ( *Will be released soon* )
  - [**To-do**] Code for building generative classifiers. ( *Will be released soon* )

Report any bug or suggestion to **tiwarilokender@gmail.com**.


