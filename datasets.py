#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 13:39:22 2020

@author: falmuqhim
"""
from torch.utils.data import Dataset
import torch
import numpy as np
import pickle
import sys


class CC200Dataset(Dataset):
    def __init__(self, pkl_filename=None, data=None, samples_list=None,
                 prog=False, regs=None, model=None, selector=None):
        self.regs = regs
        self.selector = selector
        if pkl_filename is not None:
            if prog:
                print('Loading ..!', end=' ')
            self.data = pickle.load(open(pkl_filename, 'rb'))
        elif data is not None:
            self.data = data.copy()
        else:
            sys.stderr.write('Eigther PKL file or data is needed!')
            return

        if prog:
            print('Preprocess..!', end='  ')
        if samples_list is None:
            self.flist = [f for f in self.data]
        else:
            self.flist = [f for f in samples_list]
        self.labels = np.array([self.data[f][1] for f in self.flist])

        current_flist = np.array(self.flist.copy())
        current_lab0_flist = current_flist[self.labels == 0]
        current_lab1_flist = current_flist[self.labels == 1]
        if prog:
            print(' Num Positive : ', len(current_lab1_flist), end=' ')
            print(' Num Negative : ', len(current_lab0_flist), end='\n')

        self.num_data = len(self.flist)

    def __getitem__(self, index):
        fname = self.flist[index]
        data = self.data[fname][0].copy()
        if self.regs is not None:
            data = data[self.regs].copy()
        if self.selector is not None:
            data = self.selector.transform([data])
            data = data.ravel()
        label = (self.labels[index],)
        return torch.FloatTensor(data), torch.LongTensor(label)

    def __len__(self):
        return self.num_data


class PairsDatasetCC200(Dataset):
    """
    Train: For each sample creates randomly a positive pair with label 1, and
    a negative pair with label 0
    Test: Creates fixed pairs for testing
    """

    def __init__(self, pkl_filename=None, data=None, samples_list=None,
                 prog=False, regs=None, test=False, model=None):
        self.regs = regs
        self.test = test
        self.model = model
        if pkl_filename is not None:
            if prog:
                print('Loading ..!', end=' ')
            self.data = pickle.load(open(pkl_filename, 'rb'))
        elif data is not None:
            self.data = data.copy()
        else:
            sys.stderr.write('Eigther PKL file or data is needed!')
            return

        if prog:
            print('Preprocess..!', end='  ')
        if samples_list is None:
            self.flist = [f for f in self.data]
        else:
            self.flist = [f for f in samples_list]

        # self.labels = np.array([get_label(f) for f in self.flist])
        self.labels = np.array([self.data[f][1] for f in self.flist])

        self.current_flist = np.array(self.flist.copy())
        current_lab0_flist = self.current_flist[self.labels == 0]
        current_lab1_flist = self.current_flist[self.labels == 1]
        if prog:
            print(' Num Positive : ', len(current_lab1_flist), end=' ')
            print(' Num Negative : ', len(current_lab0_flist), end=' ')

        if self.test:
            random_state = np.random.RandomState(29)
            self.current_flist = np.array(self.flist.copy())
            pairs = []
            targets = []
            for i in range(len(self.flist)):
                positive_pairs = self.current_flist[self.labels ==
                                                    self.labels[i]]
                negative_pairs = self.current_flist[self.labels !=
                                                    self.labels[i]]
                anchor = self.flist[i]
                positive = anchor
                while anchor == positive:
                    positive = random_state.choice(positive_pairs)
                negative = random_state.choice(negative_pairs)
                pairs.append([anchor, positive, negative])
                targets.append([0, 1])
                # pairs.append([anchor, negative])
                # targets.append([0])
            self.test_pairs = pairs
            self.test_labels = targets
            self.num_data = len(self.train_pairs)
            if prog:
                print(' test_pairs : ', len(self.test_pairs), end=' ')
                print(' test_labels : ', len(self.test_labels), end=' ')
        else:
            random_state = np.random.RandomState(29)
            self.current_flist = np.array(self.flist.copy())
            pairs = []
            targets = []
            for i in range(len(self.flist)):
                positive_pairs = self.current_flist[self.labels ==
                                                    self.labels[i]]
                negative_pairs = self.current_flist[self.labels !=
                                                    self.labels[i]]
                anchor = self.flist[i]
                positive = anchor
                while anchor == positive:
                    positive = random_state.choice(positive_pairs)
                negative = random_state.choice(negative_pairs)
                pairs.append([anchor, positive, negative])
                targets.append([0, 1])
                # pairs.append([anchor, negative])
                # targets.append([0])
            self.train_pairs = pairs
            self.train_label = targets
            self.num_data = len(self.train_pairs)
            if prog:
                print(' test_pairs : ', len(self.train_pairs), end=' ')
                print(' test_labels : ', len(self.train_label), end=' ')

    def __getitem__(self, index):
        if self.test:
            x1 = self.test_pairs[index][0]
            x1 = self.data[x1][0].copy()
            if self.regs is not None:
                x1 = x1[self.regs].copy()

            x2 = self.test_pairs[index][1]
            x2 = self.data[x2][0].copy()
            if self.regs is not None:
                x2 = x2[self.regs].copy()

            x3 = self.test_pairs[index][1]
            x3 = self.data[x3][0].copy()
            if self.regs is not None:
                x3 = x3[self.regs].copy()

            label = self.test_labels[index]
        else:
            x1 = self.train_pairs[index][0]
            x1 = self.data[x1][0].copy()
            if self.regs is not None:
                x1 = x1[self.regs].copy()

            x2 = self.train_pairs[index][1]
            x2 = self.data[x2][0].copy()
            if self.regs is not None:
                x2 = x2[self.regs].copy()

            x3 = self.train_pairs[index][1]
            x3 = self.data[x3][0].copy()
            if self.regs is not None:
                x3 = x3[self.regs].copy()

            label = self.train_label[index]

        x1 = torch.FloatTensor(x1)
        x2 = torch.FloatTensor(x2)
        x3 = torch.FloatTensor(x3)
        label = torch.FloatTensor(label)
        # if self.model is not None:
        #     self.model = self.model.to(Config.device)
        #     x1 = x1.to(Config.device)
        #     x2 = x2.to(Config.device)
        #     x1 = self.model.fc_encoder(x1)
        #     x2 = self.model.fc_encoder(x2)

        return [x1, x2, x3], label

    def __len__(self):
        return self.num_data
