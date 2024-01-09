#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 12:27:15 2020

@author: falmuqhim
"""
import torch


class Config():
    data_path = './data/'
    phenotypic_file = 'Phenotypic_V1_0b_preprocessed1.csv'
    n_fold = 5
    center = 'NYU'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 16
    num_epochs = 30
    learning_rate = 1e-3
    margin = 1.0
    load_AE_model = False
    lemda = 1e-5
    beta = 2
    p = 0.05
    iterations = 10
