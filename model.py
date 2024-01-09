#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 20:48:16 2020

@author: falmuqhim
"""

import torch.nn as nn
import torch
import torch.nn.functional as F


class SAE(nn.Module):
    def __init__(self, n_features, n_lat, n_classes=2):
        super().__init__()
        self.encoder = nn.Linear(n_features, n_lat)
        self.decoder = nn.Linear(n_lat, n_features)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(n_lat),
            nn.Dropout(0.8),
            nn.Linear(n_lat, int(n_lat/2)),
            nn.BatchNorm1d(int(n_lat/2)),
            nn.Dropout(0.8),
            nn.Linear(int(n_lat/2), 500),
            nn.BatchNorm1d(500),
            nn.Dropout(0.9),
            nn.Linear(500, n_classes),
        )

    def forward(self, x, classify=True):
        x = self.encoder(x)
        x = F.relu(x)
        if classify:
            target = self.classifier(x)
        else:
            target = None
        x = self.decoder(x)
        return x, target


class NetSNN(nn.Module):
    def __init__(self, n_features, n_lat, n_classes=2):
        super().__init__()

        self.encoder = nn.Linear(n_features, n_lat)
        self.decoder = nn.Linear(n_lat, n_features)

        # self.linear1 = nn.Linear(n_lat, 2000)
        # self.linear2 = nn.Linear(2000, 500)
        self.snn = nn.Sequential(
            # nn.BatchNorm1d(n_lat),
            # nn.Dropout(0.8),
            nn.Linear(n_lat, int(n_lat/2)),
            # nn.BatchNorm1d(int(n_lat/2)),
            # nn.Dropout(0.8),
            nn.Tanh(),
            nn.Linear(int(n_lat/2), 500),
            nn.Tanh()
        )
        self.diff = nn.Sequential(
            # nn.BatchNorm1d(500),
            # nn.Dropout(0.9),
            nn.Linear(500, n_classes),
        )
        # self.linear3 = nn.Linear(500, n_classes)

    def forward(self, data):
        reco = self.encoder(data[0])
        res = []
        for i in range(2):  # Siamese nets; sharing weights
            x = data[i]
            x = self.encoder(x)
            x = torch.tanh(x)
            x = self.snn(x)
            res.append(x)
        output = torch.abs(res[1] - res[0])
        output = self.diff(output)
        reco = self.decoder(reco)
        return reco, output
