   
![alt text](https://github.com/lokender/REGroup/blob/master/REGroup_teaser.png "REGroup Teaser")

This repository contains the demo code of the method called **REGroup** proposed in the paper: *Dissecting Deep Networks into an Ensemble of GenerativeClassifiers for Robust Predictions*. The **REGroup**, stands for *Rank-aggregating Ensemble of Generative Classifiers for Robust Predictions*. 

If you use this repository or REGroup in your research/product, please please consider citing:.
   
      @article{tiwari2020dissecting,
        title={Dissecting Deep Networks into an Ensemble of Generative Classifiers for Robust Predictions},
        author={Tiwari, Lokender and Madan, Anish and Anand, Saket and Banerjee, Subhashis },
        journal={arXiv preprint arXiv:2006.10679},
        year={2020}
      }

# Requirements
  - Pytorch 
  - numpy, scipy 
  - matplotlib 
  - Jupyter notebook 
  - foolbox (version 2.3.0)
  


# Steps to run the demo
- Clone the repository.
- [Download](https://drive.google.com/file/d/1ylJctBJzh4ih-0zzD4ZLO2umh--QpX7u/view?usp=sharing) CIFAR10 PGD L-infinity adversarial examples 
- Open jupyter notebook **REGroup_demo_cifar10_vgg19.ipynb**

```sh
$ git clone https://github.com/lokender/REGroup.git
$ cd REGroup
$ wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ylJctBJzh4ih-0zzD4ZLO2umh--QpX7u' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1ylJctBJzh4ih-0zzD4ZLO2umh--QpX7u" -O cifar10_vgg19_pgd_examples.mat && rm -rf /tmp/cookies.txt
```

# To-dos?
  - [**Done**] **Classifier**: VGG19, **Dataset**: CIFAR10  ( *Released* )
  - [**To-do**] **Classifier**: VGG19, **Dataset**: ImageNet ( *Will be released soon* )
  - [**To-do**] **Classifier**: ResNet, **Dataset**: CIFAR10 ( *Will be released soon* )
  - [**To-do**]  **Classifier**: ResNet, **Dataset**: ImageNet ( *Will be released soon* )
  - [**To-do**] **Classifier**: Inception, **Dataset**: ImageNet ( *Will be released soon* )
  - [**To-do**] Code for building generative classifiers. ( *Will be released soon* )

Report any bug or suggestion to **tiwarilokender@gmail.com**.

## Press STAR on the top right of this page for continuous updates.


