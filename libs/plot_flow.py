#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 19:13:07 2019

@author: HIT data mining lab
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from sklearn import metrics
from IPython.core.pylabtools import figsize  # import figsize
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from scipy import interp
from sklearn.preprocessing import label_binarize


# %%最好的loss作图
def plot_loss(history):
    pd.options.display.float_format = '{:.1f}'.format
    sns.set()  # Default seaborn look and feel
    plt.style.use('default')
    print("\n--- Learning curve of model training ---\n")

    # summarize history for accuracy and loss
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(history.history['acc'], "g--", label="Accuracy of training data")
    plt.plot(history.history['val_acc'], "g", label="Accuracy of validation data")
    plt.plot(history.history['loss'], "r--", label="Loss of training data")
    plt.plot(history.history['val_loss'], "r", label="Loss of validation data")
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 18,
             'color': 'black'
             }
    plt.title('G2_Model Accuracy and Loss')
    plt.ylabel('Accuracy and Loss', font2)
    plt.xlabel('Training Epoch', font2)
    plt.ylim(0)

    # plt.legend()
    plt.legend(loc=0, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize='12')

    plt.tick_params(labelsize=12, labelcolor='black')
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    # ax.tick_params(axis='both',colors='black')

    figsize(6, 4)  # 设置 figsize
    plt.rcParams['savefig.dpi'] = 300  # 图片像素
    plt.rcParams['figure.dpi'] = 300  # 分辨率
    fig = plt.gcf()  # 获得该图的句柄
    plt.show()
    if not os.path.exists('./figures/'):
        os.makedirs('./figures/')
    fig.savefig('./figures/G2_G1_pre_loss.png', dpi=300, bbox_inches='tight')  # 指定分辨率保存


# %% 最好的画混淆矩阵的图
def plot_confusion_matrix(truelabel, predictlabel, titlename, figname):
    # compute evaluation metrics
    matrix = metrics.confusion_matrix(truelabel, predictlabel)

    # plot and save confusion matrix heat map
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    h = sns.heatmap(matrix,
                    cmap="coolwarm",
                    linecolor='white',
                    linewidths=1,
                    xticklabels=['Normal0', 'Fault1', 'Fault2', 'Fault3', 'Fault4'],
                    yticklabels=['Normal0', 'Fault1', 'Fault2', 'Fault3', 'Fault4'],
                    ax=ax1,
                    cbar=False,
                    annot=True,
                    annot_kws={'size': 12, 'weight': 'bold', 'color': 'black'},
                    fmt="d")
    cb = h.figure.colorbar(h.collections[0])  # 显示colorbar
    cb.ax.tick_params(labelsize=12, labelcolor='black')  # 设置colorbar刻度字体大小。
    ax1.set_xticklabels(['Normal0', 'Fault1', 'Fault2', 'Fault3', 'Fault4'],
                        rotation=0, fontsize=12, color='black')
    ax1.set_yticklabels(['Normal0', 'Fault1', 'Fault2', 'Fault3', 'Fault4'],
                        rotation=0, fontsize=12, color='black')
    #  ax.tick_params(labelsize=10,color='black')
    plt.rcParams['savefig.dpi'] = 300  # 图片像素
    plt.rcParams['figure.dpi'] = 300  # 分辨率

    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 12,
             'color': 'black'
             }
    plt.title(titlename, fontsize=12)
    plt.ylabel("True Label", font2)
    plt.xlabel("Predicted Label", font2)
    fig = plt.gcf()  # 获得该图的句柄
    if not os.path.exists('./figures/'):
        os.makedirs('./figures/')
    fig.savefig('./figures/' + figname, dpi=300, bbox_inches='tight')  # 指定分辨率和路径保存


# 最好的ROC曲线绘图
def plot_ROC(x_test, y_test, finemodel, nb_classes=5):
    # %%输入数据接口
    X_valid = x_test
    Y_valid = y_test
    Y_pred = finemodel.predict(X_valid)
    Y_pred = [np.argmax(y) for y in Y_pred]  # 取出y中元素最大值所对应的索引
    Y_valid = [np.argmax(y) for y in Y_valid]

    # Binarize the output
    Y_valid = label_binarize(Y_valid, classes=[i for i in range(nb_classes)])
    Y_pred = label_binarize(Y_pred, classes=[i for i in range(nb_classes)])

    # micro：多分类　　
    # weighted：不均衡数量的类来说，计算二分类metrics的平均
    # macro：计算二分类metrics的均值，为每个类给出相同权重的分值。
    precision = metrics.precision_score(Y_valid, Y_pred, average='micro')
    recall = metrics.recall_score(Y_valid, Y_pred, average='micro')
    f1_score = metrics.f1_score(Y_valid, Y_pred, average='micro')
    accuracy_score = metrics.accuracy_score(Y_valid, Y_pred)
    print("Precision_score:", precision)
    print("Recall_score:", recall)
    print("F1_score:", f1_score)
    print("Accuracy_score:", accuracy_score)

    # roc_curve:真正率（True Positive Rate , TPR）或灵敏度（sensitivity）
    # 横坐标：假正率（False Positive Rate , FPR）

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(nb_classes):
        fpr[i], tpr[i], _ = roc_curve(Y_valid[:, i], Y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(Y_valid.ravel(), Y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(nb_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(nb_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= nb_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw = 2
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(1, 1, 1)
    l1, = plt.plot(fpr["micro"], tpr["micro"],
                   label='micro-average ROC curve (area = {0:0.2f})'
                         ''.format(roc_auc["micro"]),
                   color='deeppink', linestyle=':', linewidth=2)

    l2, = plt.plot(fpr["macro"], tpr["macro"],
                   label='macro-average ROC curve (area = {0:0.2f})'
                         ''.format(roc_auc["macro"]),
                   color='navy', linestyle='-.', linewidth=2)

    plt.tick_params(labelsize=12, labelcolor='black')
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 18,
             'color': 'black'
             }
    plt.xlabel('False Positive Rate', font2)
    plt.ylabel('True Positive Rate', font2)
    plt.title('Finetune ROC To Multi-class', fontsize=20)
    plt.legend(loc="lower right")
    fig = plt.gcf()  # 获得该图的句柄
    plt.show()
    if not os.path.exists('./figures/'):
        os.makedirs('./figures/')
    fig.savefig("./figures/ROC_20%Finetune分类.png", dpi=300, bbox_inches='tight')
