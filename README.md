<p align="center">
  <a href="https://github.com/dhyan1999/CASIA-B-GaitAnalysis" title="Gait Analysis">
  </a>
</p>
<h1 align="center"> Gait Cycle analysis using Pattern Recognition </h1>

![experiment](image/experiment.png)
<p align="center">2021G COMP-5112-GDF - Research Methodolody Computer Science</p>

<h2 align="center">🌐 Links 🌐</h2>
<p align="center">
    <a href="https://github.com/dhyan1999/CASIA-B-GaitAnalysis" title="Gait Analysis">📂 Repo</a>
    ·
    <a href="https://github.com/dhyan1999/CASIA-B-GaitAnalysis/blob/main/Paper/PR_Proposal.pdf" title="Gait Analysis">📄 Paper</a>
</p>

Abstract : One of the most significant human characteristics is motion ability, which includes gait as the foundation of human transitional movement. 
Many academics had concentrated on this topic in order to consider a novel recognition system. Many human gait datasets have been developed in the previous ten years. 
The Gait Dataset of the University of South Florida (USF), the Gait Dataset of the Chinese Academy of Sciences (CASIA), and the Gait Dataset of Southampton University (SOTON) are some of the most extensively utilised datasets.
The CASIA Gait Dataset will be examined in this research to determine its properties.Gait patterns were gathered in this investigation utilising a wireless platform with two sensors attached to the individuals’ chest and right ankle.
The raw data was then subjected to certain preprocessing techniques.The performance of many temporal and frequency domain features is evaluated using five different classifiers, and a full comparison is made in this work.

## Table of Content

1. [Manifest](#-manifest)
2. [Prerequisites](#-prerequisites)
3. [Implementation of Code](##-implementation-of-code)
4. [Future scope](#-future-scope)
5. [Video](#video)

## 🧑🏻‍🏫 Manifest

```
- Pre_Processing.ipynb --> A python .ipynb file that run's Pre processing of the dataset
- Resnet50.ipynb --> A python .ipynb file that run's ResNet50 Model
- Resnet152V2.ipynb --> A python .ipynb file that run's Resnet152V2 Model
- InceptionV2.ipynb --> A python .ipynb file that run's InceptionV2 Model
- InceptionV3.ipynb --> A python .ipynb file that run's InceptionV3 Model
- Paper   --> A Detailed research paper on Gait Cycle analysis using Pattern Recognition
- Requierments.txt --> A requierment file for all kind of libraries to be requiered
- README.md ---> This markdown file you are reading.
```


## 🤔 Prerequisites

- [Python](https://www.python.org/ "Python") Installed

- Python Basics Understanding

- Understanding of Machine Learning and Deep Learning libraries

- A deep knowledge of [Transfer Learning models](https://keras.io/api/applications/)

## 👨🏻‍💻 Implementation of Code

Resnet50

```py
res = ResNet50(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
```
![resnet50](image/resnet50.png)

