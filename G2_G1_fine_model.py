#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 19:13:07 2019

@author: HIT data mining lab
"""

# transfer G2 turbines(old) to G1 turbien(new) & finetuned experiments

import tensorflow as tf
import keras
from algorithms.CNN_Module import cnn_model
from libs.plot_flow import plot_loss, plot_confusion_matrix, plot_ROC

from keras import optimizers
from keras.optimizers import Adam, SGD
from keras.callbacks import Globalbest
from keras.utils import np_utils

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing as skpp  # 加入sklearn归一化操作
from sklearn.model_selection import train_test_split

from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
import dill
import time
import scipy.io

# define parameters
n_classes = 5  # 0-9 digits
n_width = 1
n_height = 7
n_depth = 1
n_inputs = n_height * n_width * n_depth  # total pixels

n_filters = [32, 32]
learning_rate = 0.005
n_epochs = 100  # epochs
batch_size = 16


def main():
    # data loadmat
    # %% 迁移微调数据集X_GE9FA_1
    X1_data = scipy.io.loadmat('../X_GE9FA_1.mat')
    X1 = X1_data['X_GE9FA_1']
    Y1_data = scipy.io.loadmat('../Y_GE9FA_1.mat')
    Y1 = Y1_data['Y_GE9FA_1']
    Xtr, Xts, ytr, yts = train_test_split(X1, Y1, test_size=0.8, stratify=Y1)

    # 分别normalize all the data (0,1)
    scaler = skpp.MinMaxScaler(feature_range=(0, 1))
    normalized_traininput = scaler.fit_transform(Xtr)
    normalized_testinput = scaler.transform(Xts)

    num_classes = 5
    y_train = np_utils.to_categorical(ytr, num_classes)
    # qiuhe=np.sum(y_train[:,1])检验标签是否正确
    y_test = np_utils.to_categorical(yts, num_classes)

    x_train = normalized_traininput
    x_test = normalized_testinput

    # build graph
    tf.reset_default_graph()
    keras.backend.clear_session()

    # load model
    finemodel = cnn_model(n_width=1, n_height=7, n_depth=1)

    # 加载模型权值从h5文件中
    finemodel.load_weights('../chapter3/G2_pretrain_weights.h5', by_name=True)
    finemodel.summary()

    # put weight probes
    # 获得某一层的权重和偏置
    weightxin_conv2d_1, biasxin_conv2d_1 = finemodel.get_layer('conv2d_1').get_weights()
    weightxin_dense_1, biasxin_dense_1 = finemodel.get_layer('new_dense1').get_weights()

    # print(weight_conv2d_1.shape)
    # print(bias_conv2d_1.shape)

    # 获取模型的层数和名字
    print("layer nums:", len(finemodel.layers))
    for layer in finemodel.layers:
        print(layer.name)

    # 冻结训练的层数，根据模型的不同，层数也不一样，根据调试的结果，
    FREEZE_LAYERS = 8
    # 除了FC层，靠近FC层的一部分卷积层可参与参数训练，
    # 一般来说，模型结构已经标明一个卷积块包含的层数，
    # 在这里我们选择FREEZE_LAYERS为14，表示最除了卷积块，FC层要参与参数训练
    for layer in finemodel.layers[:FREEZE_LAYERS]:
        layer.trainable = False
    for layer in finemodel.layers[FREEZE_LAYERS:]:
        layer.trainable = True
    # 查看冻结之后训练的层
    for layer in finemodel.layers:
        print("layer.trainable:", layer.trainable)
    # 查看冻结之后训练的参数
    finemodel.summary()

    # 查看可训练(trainable)和不可训练(non_trainable)的权值方法：
    print('参与训练的权值名称：')
    for x in finemodel.trainable_weights:
        print(x.name)
    print('\n')

    print('不参与训练的权值名称：')
    for x in finemodel.non_trainable_weights:
        print(x.name)
    print('\n')

    # 训练新的finemodel
    globalbest = Globalbest(monitor='val_loss', epochs=n_epochs)
    sgd = optimizers.SGD(learning_rate, decay=0.001, momentum=0.9, nesterov=False)
    finemodel.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])  # optimizer=SGD(lr=learning_rate)

    history = finemodel.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=n_epochs,
                            validation_split=0.1,
                            verbose=1,
                            callbacks=[globalbest])

    y_pre = finemodel.predict(x_test)
    predictlabel = np.argmax(y_pre, axis=1)  # predicted label of test data

    # score中含有两个元素，第0个元素为交叉熵损失函数，第1个元素为classification accuracy
    score = finemodel.evaluate(x_test, y_test, batch_size=32)
    print('\nTest loss:', score[0])
    print('Test accuracy:', score[1])
    plt.plot(predictlabel, '.')
    plt.show()
    # plot loss
    plot_loss(history)
    # plot confusion matrix
    plot_confusion_matrix(yts, predictlabel, 'Finetune', 'finetune')
    # plot ROC curve
    plot_ROC(x_test, y_test, finemodel, nb_classes=5)


if __name__ == '__main__':
    exit(main())

# 只用20%的数据微调就这个效果了
# Test loss: 0.10763418752079208
# Test accuracy: 0.9645833333333333
# 冻结前八层卷积层，学习率0.01，bachsize16，decay=0.01
# 只用cnn网络，20%数据
# Test loss: 0.39643581795195737
# Test accuracy: 0.8723958333333334
# 正好说明数据集0和数据1MMD距离大，所以样本差异较大，这样微调加大学习率0.01，50迭代即可
# 数据集0和数据集2MMD距离较小，更适合微调，只需要很小的学习率，哪怕固定一些层的参数特征，也能达到很好的效果。
