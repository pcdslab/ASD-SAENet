#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 11:11:25 2020

@author: falmuqhim
"""

from config import Config
from model import SAE
from datasets import CC200Dataset

import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import sys
import time
from torch.utils.data import DataLoader
import torch
import pyprind
import pickle
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
import numpy.ma as ma  # for masked arrays
from datetime import datetime
import gc
import matplotlib
import matplotlib.pyplot as plt
import argparse
from scipy import interp
from sklearn.feature_selection import RFE, SelectKBest, SelectFdr, f_classif
from sklearn.svm import SVR
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score


# Global Variables
labels = {}


def get_key(filename):
    f_split = filename.split('_')
    if f_split[3] == 'rois':
        key = '_'.join(f_split[0:3])
    else:
        key = '_'.join(f_split[0:2])
    return key


def get_label(filename):
    assert (filename in labels)
    return labels[filename]


def get_corr_data_dynamic(filename, windLength, stepSize):
    for file in os.listdir(Config.data_path):
        if file.startswith(filename):
            brainData = np.loadtxt(
                open(os.path.join(Config.data_path, file), "rb"), delimiter="\t")
    leftStile, rightStile = 0, stepSize
    dataParts = []
    while rightStile <= windLength:
        dataParts.append(brainData[leftStile:rightStile, 0:])
        leftStile = leftStile + stepSize
        rightStile = rightStile + stepSize

    data = []
    for each in dataParts:
        with np.errstate(invalid="ignore"):
            corr = np.corrcoef(each.T)
            mask = np.invert(np.tri(corr.shape[0], k=-1, dtype=bool))
            m = ma.masked_where(mask == 1, mask)
            final1D = ma.masked_where(m, corr).compressed()
            data.extend(final1D)
    return data


def get_corr_data(filename):
    for file in os.listdir(Config.data_path):
        if file.startswith(filename):
            df = pd.read_csv(os.path.join(Config.data_path, file), sep='\t')

    with np.errstate(invalid="ignore"):
        corr = np.nan_to_num(np.corrcoef(df.T))
        mask = np.invert(np.tri(corr.shape[0], k=-1, dtype=bool))
        m = ma.masked_where(mask == 1, mask)
        return ma.masked_where(m, corr).compressed()


def confusion(g_turth, predictions):
    tn, fp, fn, tp = confusion_matrix(g_turth, predictions).ravel()
    accuracy = (tp+tn)/(tp+fp+tn+fn)
    sensitivity = (tp)/(tp+fn)
    specificty = (tn)/(tn+fp)
    return accuracy, sensitivity, specificty


def get_selector_RFE(all_corr, samplesnames, regions, n_features):
    X = []
    y = []
    for sn in samplesnames:
        X.append(all_corr[sn][0][regions])
        y.append(all_corr[sn][1])
    svm = LinearSVC()
    rfe = RFE(svm, n_features_to_select=n_features)
    rfe = rfe.fit(X, y)
    return rfe


def get_selector_FDR(all_corr, samplesnames, alpha):
    X = []
    y = []
    for sn in samplesnames:
        X.append(all_corr[sn][0])
        y.append(all_corr[sn][1])
    X = pd.DataFrame(X)
    X = X.fillna(X.mean())
    X = X.values.tolist()
    fdr = SelectFdr(f_classif, alpha=alpha)
    fdr = fdr.fit(X, y)
    return fdr


def get_selector_KBest(all_corr, samplesnames, k):
    X = []
    y = []
    for sn in samplesnames:
        X.append(all_corr[sn][0])
        y.append(all_corr[sn][1])
    X = pd.DataFrame(X)
    X = X.fillna(X.mean())
    X = X.values.tolist()
    kbest = SelectKBest(f_classif, k=k)
    kbest = kbest.fit(X, y)
    return kbest


def get_regs(all_corr, samplesnames, regnum):
    datas = []
    for sn in samplesnames:
        datas.append(all_corr[sn][0])
    datas = np.array(datas)
    avg = []
    for ie in range(datas.shape[1]):
        avg.append(np.mean(datas[:, ie]))
    avg = np.array(avg)
    highs = avg.argsort()[-regnum:][::-1]
    lows = avg.argsort()[:regnum][::-1]
    regions = np.concatenate((highs, lows), axis=0)
    return regions


def get_loader(pkl_filename=None, data=None, samples_list=None, batch_size=64, num_workers=1, mode='train', prog=False, regions=None, model=None, selector=None):
    """Build and return data loader."""
    if mode == 'train':
        shuffle = True
    else:
        shuffle = False

    dataset = CC200Dataset(pkl_filename=pkl_filename, data=data, samples_list=samples_list,
                           prog=prog, regs=regions, model=model, selector=selector)

    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers, drop_last=True)

    return data_loader


def kl_divergence(rho, rho_hat):
    rho_hat = torch.mean(torch.sigmoid(rho_hat), 1)
    rho = torch.tensor([rho] * len(rho_hat)).to(Config.device)
    return torch.sum(rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log(
        (1 - rho) / (1 - rho_hat)))


def sparse_loss_kl(model, images):
    model_children = list(model.children())
    loss = 0
    values = images
    for i in range(len(model_children) - 1):
        values = (model_children[i](values))
        loss += kl_divergence(Config.p, values)
    return loss


def freez_layer(model):
    for name, param in model.named_parameters():
        if name.startswith('encoder'):
            param.requires_grad = False


def train(model, train_loader, loss_ae, loss_clf, optimizer, mode='both'):
    epoch_loss = 0
    count = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        count += len(target)
        data = data.to(Config.device)
        target = target.to(Config.device)
        target = target.squeeze(1)
        if mode != 'both':
            optimizer.zero_grad()
            output, target_hat = model(data)
            loss = loss_clf(target_hat, target)
        else:
            optimizer.zero_grad()
            output, target_hat = model(data)
            loss = loss_ae(output, data) / len(data)
            loss += loss_clf(target_hat, target)
            loss += Config.beta * sparse_loss_kl(model, data)
        loss.backward()
        optimizer.step()
        epoch_loss += float(loss)
    epoch_loss = epoch_loss/count
    gc.collect()
    torch.cuda.empty_cache()
    return epoch_loss


def test(model, test_loader, loss_ae, loss_clf):
    true_count = 0
    count = 0
    y_true = []
    all_predss = []
    all_prob = []
    model.eval()
    with torch.no_grad():
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.to(Config.device)
            target = target.to(Config.device)
            output, target_hat = model(data)
            loss = loss_ae(output, data)
            epoch_loss += float(loss)
            target = target.detach().cpu().numpy()
            count += len(target)
            y_arr = np.array(target, dtype=np.int32)
            y_true.extend(y_arr.tolist())

            probs = target_hat.detach().cpu().numpy()
            all_prob.extend(probs)
            # for softmax 2 classes
            target_hat = torch.argmax(
                target_hat, dim=1).detach().cpu().numpy()
            all_predss.extend(target_hat)
            true_count += np.sum(target_hat == y_arr)
        if count != 0:
            epoch_loss = epoch_loss/count

        acc, sens, spef = confusion(y_true, all_predss)
        # results = open('macro_auc.txt', 'a')
        # all_prob = np.array(all_prob)
        # fpr = dict()
        # tpr = dict()
        # roc_auc = dict()
        # for i in range(2):
        #     fpr[i], tpr[i], _ = roc_curve(y_true, all_prob[:, i])
        #     roc_auc[i] = auc(fpr[i], tpr[i])
        # # First aggregate all false positive rates
        # all_fpr = np.unique(np.concatenate([fpr[i] for i in range(2)]))

        # # Then interpolate all ROC curves at this points
        # mean_tpr = np.zeros_like(all_fpr)
        # for i in range(2):
        #     mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # # Finally average it and compute AUC
        # mean_tpr /= 2

        # fpr["macro"] = all_fpr
        # tpr["macro"] = mean_tpr
        # roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # # fpr, tpr, _ = roc_curve(y_true, all_predss)
        # # auc = roc_auc_score(y_true, all_predss)
        # f = ','.join(str(e) for e in fpr["macro"])
        # t = ','.join(str(e) for e in tpr["macro"])
        # results.write('fpr' + ',' + f + '\n')
        # results.write('tpr' + ',' + t + '\n')
        # results.write('auc' + ',' + str(roc_auc["macro"]) + '\n')
        # results.close()
    gc.collect()
    torch.cuda.empty_cache()
    return acc, sens, spef


def calculate_accuracy(y_true, y_pred):
    predicted = y_pred.ge(.5).view(-1)
    return (y_true == predicted).sum().float() / len(y_true)


def main():
    print('GPU available: ', (Config.device == 'cuda'))
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--center', type=str, default=None,
                    help='what center to train and test')
    ap.add_argument('-e', '--epochs', type=int, default=25,
                    help='number of epochs to train our network for')
    ap.add_argument('-f', '--folds', type=int, default=10,
                    help='number of folds to train our network with')
    ap.add_argument('-r', '--result', type=int, default=0,
                    help='would you like to write the result in file, 0 for no')
    ap.add_argument('-i', '--iter', type=int, default=10,
                    help='number of iterations')
    ap.add_argument('-p', '--pretrain', type=int, default=0,
                    help='epochs to pre-train the model')
    # ap.add_argument('-s', '--selector', type=int, default=0,
    #                 help='1 for RFE, 2 for FDR, and 0 for none')
    ap.add_argument('-a', '--alpha', type=float, default=0.5,
                    help='alpha value for the FDR selector')

    ap.add_argument('-d', '--dynamic', type=int, default=0,
                    help='use dynamic correlation, 0 for no otherwise yes')

    args = vars(ap.parse_args())
    center = args['center']
    epochs = args['epochs']
    folds = args['folds']
    out = args['result']
    iterations = args['iter']
    pretrain = args['pretrain']
    # sel = args['selector']
    alpha = args['alpha']
    dynamic = args['dynamic']
    output_name = 'results.csv'
    if out != 0:
        results = open(output_name, 'a')
        print('Result will written in {0}'.format(output_name))
        results.write(',' + ' '.join(sys.argv[1:]) + 'beta: ' +
                      str(Config.beta) + ', p: ' + str(Config.p) + '\n')
        results.close()
    if center is not None:
        print('Starting center: ' + center)
    else:
        print('Starting all centers ...')
    gc.collect()
    torch.cuda.empty_cache()
    flist = [f for f in os.listdir(Config.data_path) if not f.startswith('.')]

    for f in range(len(flist)):
        flist[f] = get_key(flist[f])

    if center is not None:
        centers_dict = {}
        for f in flist:
            key = f.split('_')[0]

            if key not in centers_dict:
                centers_dict[key] = []
            centers_dict[key].append(f)

        flist = np.array(centers_dict[center])

    df_labels = pd.read_csv('./' + Config.phenotypic_file)

    df_labels.DX_GROUP = df_labels.DX_GROUP.map({1: 1, 2: 0})

    for row in df_labels.iterrows():
        file_id = row[1]['FILE_ID']
        y_label = row[1]['DX_GROUP']
        if file_id == 'no_filename':
            continue
        assert(file_id not in labels)
        labels[file_id] = y_label

    if dynamic == 0:
        if not os.path.exists('./correlations_file.pkl'):
            pbar = pyprind.ProgBar(len(flist))
            all_corr = {}
            for f in flist:
                lab = get_label(f)
                all_corr[f] = (get_corr_data(f), lab)
                pbar.update()

            print('Corr-computations finished')

            pickle.dump(all_corr, open('./correlations_file.pkl', 'wb'))
            print('Saving to file finished')

        else:
            all_corr = pickle.load(open('./correlations_file.pkl', 'rb'))
    else:
        # this is just to try if dynamic window would make sense.. plz disregard
        if not os.path.exists('./correlations_file_dynamic.pkl'):
            pbar = pyprind.ProgBar(len(flist))
            all_corr = {}
            for f in flist:
                lab = get_label(f)
                all_corr[f] = (get_corr_data_dynamic(f, 144, 36), lab)
                pbar.update()

            print('Corr-computations finished')

            pickle.dump(all_corr, open(
                './correlations_file_dynamic.pkl', 'wb'))
            print('Saving to file finished')
        else:
            all_corr = pickle.load(
                open('./correlations_file_dynamic.pkl', 'rb'))
        X = []
        # y = []
        for index, data in all_corr.items():
            X.append(data[0])
            # y.append(data[1])
        X = pd.DataFrame(X)
        X = X.fillna(X.mean())
        X = X.values.tolist()
        i = 0
        for index, data in all_corr.items():
            all_corr[index] = (X[i], data[1])
            i += 1
    num_corr = len(all_corr[flist[0]][0])
    random_state = np.random.RandomState(29)

    y_arr = np.array([get_label(f) for f in flist])
    flist = np.array(flist)
    kk = 0
    n_features = 5000
    n_lat = 2000
    accuracies_iter = []
    overall_result = []
    total_time = 0
    for i in range(iterations):
        # if center == None:
        kf = StratifiedKFold(
            n_splits=folds, random_state=random_state, shuffle=True)
        np.random.shuffle(flist)
        # else:
        # kf = StratifiedKFold(n_splits=folds)
        # np.random.shuffle(flist)
        y_arr = np.array([get_label(f) for f in flist])
        accurcies_fold = []
        res = []
        print('Entering iteration: ' + str(i+1))
        start = time.time()
        for kk, (train_index, test_index) in enumerate(kf.split(flist, y_arr)):
            print('---------------------------')
            now = datetime.now()
            print('Entering ' + str(kk+1) + ' Fold')
            train_samples, test_samples = flist[train_index], flist[test_index]
            prog = (True if (kk == 0) else False)

            # if sel == 1:
            #     regions_inds = get_regs(
            #         all_corr, train_samples, int(num_corr/4))
            #     selector = get_selector_RFE(
            #         all_corr, train_samples, regions_inds, n_features)
            # elif sel == 2:
            #     selector = get_selector_KBest(
            #         all_corr, train_samples, 3000)
            #     n_features = 3000
            #     regions_inds = None
            # else:
            regions_inds = regions_inds = get_regs(
                all_corr, train_samples, int(num_corr/4))
            selector = None
            n_features = len(regions_inds)

            train_loader = get_loader(data=all_corr, samples_list=train_samples,
                                      batch_size=Config.batch_size, mode='train',
                                      prog=prog, regions=regions_inds, selector=selector)

            test_loader = get_loader(data=all_corr, samples_list=test_samples,
                                     batch_size=Config.batch_size, mode='test',
                                     prog=prog, regions=regions_inds, selector=selector)

            # Build AutoEncoder model for feature extractions
            model = SAE(n_features, int(n_features/2), 2)
            model = model.to(Config.device)
            criterion_ae = nn.MSELoss()
            criterion_ae = criterion_ae.to(Config.device)
            criterion_clf = nn.CrossEntropyLoss()
            # criterion_clf = nn.BCEWithLogitsLoss()
            criterion_clf = criterion_clf.to(Config.device)
            optimizer = torch.optim.Adam([{'params': model.encoder.parameters(), 'lr': Config.learning_rate,
                                           'weight_decay': 1e-4},
                                          #   {'params': model.encoder2.parameters(), 'lr': Config.learning_rate,
                                          #    'weight_decay': 1e-5},
                                          {'params': model.decoder.parameters(), 'lr': Config.learning_rate,
                                           'weight_decay': 1e-4},
                                          {'params': model.classifier.parameters(), 'lr': 0.001,
                                           'weight_decay': 0.1}])
            pbar = pyprind.ProgBar(epochs)
            for epoch in range(epochs):
                if pretrain != 0:
                    if epoch < pretrain:
                        train(model, train_loader, criterion_ae,
                              criterion_clf, optimizer)
                    else:
                        train(model, train_loader, criterion_ae,
                              criterion_clf, optimizer, 'clf')
                else:
                    train(model, train_loader, criterion_ae,
                          criterion_clf, optimizer)
                pbar.update()
            acc, sens, spef = test(
                model, test_loader, criterion_ae, criterion_clf)
            res.append([acc, sens, spef])
        finish = time.time()
        total_time += (finish-start)
        if out != 0:
            results = open(output_name, 'a')
            r = np.mean(res, axis=0).tolist()
            if center is not None:
                results.write('repeat,' + str(i+1) + ',' +
                              str(r[0]) + ',' + str(r[1]) + ',' + str(r[2]) + ',' + center + '\n')
            else:
                results.write('repeat,' + str(i+1) + ',' +
                              str(r[0]) + ',' + str(r[1]) + ',' + str(r[2]) + ',' + 'all' + '\n')
            results.close()
        else:
            print("repeat: ", (i+1), np.mean(res, axis=0).tolist())
        overall_result.append(np.mean(res, axis=0).tolist())

    if out != 0:
        results = open(output_name, 'a')
        r = np.mean(overall_result, axis=0).tolist()
        if center is not None:
            results.write('overall,' + str(10) + ',' +
                          str(r[0]) + ',' + str(r[1]) + ',' + str(r[2]) + ',' + center + '\n')
        else:
            results.write('overall,' + str(10) + ',' +
                          str(r[0]) + ',' + str(r[1]) + ',' + str(r[2]) + ',' + 'all' + '\n')
        results.close()
    else:
        print("---------------Result of repeating 10 times-------------------")
        print(np.mean(np.array(overall_result), axis=0).tolist())
    if out != 0:
        results = open(output_name, 'a')
        if center is not None:
            results.write('Average Time,' + str(total_time /
                                                iterations) + ',' + center + '\n')
        else:
            results.write('Average Time,' + str(total_time /
                                                iterations) + ',' + 'all' + '\n')
        results.close()
    else:
        print("---------------Average time of executing each iteration-------------------")
        print((total_time/iterations))


if __name__ == "__main__":
    main()
