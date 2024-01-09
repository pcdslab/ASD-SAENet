# ASD-SAENet

This repository contains the implementation of ASD-SAENet algorithm.

# Research article

Fahad Almuqhim, and Fahad Saeed (2021) **ASD-SAENet: Sparse Autoencoder for detecting Autism Spectrum Disorder (ASD) using fMRI data**
**Under review**

# Enviroment Setup

## Hardware requirements

- A server containing CUDA enabled GPU with compute capability 3.5 or above.

## Software requirements

- Python version 3.7 or above
- Pytorch version 1.5.0
- CUDA version 10 or above
## Dataset
The dataset used is pre-processed, and it can be downloaded from here:
http://preprocessed-connectomes-project.org/abide/

# Parameter setting
- folds: the k value for k-fold cross-validation
- iter: number of iteration to run the training, and testing
- epochs: number of epochs to train the model
- pretrain: number of iterations to pre-tarin the SAE with the classifier before fine-tuning the classifier.
- center: which center to run, if None is given, the whole dataset will be the input.
- result: 1 to write the results in a file, 0 for not
- Example:
````
python main.py --folds=5 --iter=10 --epochs=30 --pretrain=20 --center='NYU' --result=1
````
