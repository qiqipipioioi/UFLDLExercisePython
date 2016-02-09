#!/usr/bin/python
#coding:utf-8
__author__ = "XunTu qiqipipioioi@qq.com"

import numpy as np
import scipy.optimize as sop
import matplotlib.pyplot as plt
import struct
import sys
sys.path.append('../sparseAutoencoder/')
sys.path.append('../softmax/')
import sparseAutoencoder
import softmax


def ReadMNISTImages(filename):
    #read images
    binfile = open(filename,'rb')
    buf = binfile.read()
    k = 0
    index = 0
    magic, numImages, numRows, numColumns = struct.unpack_from('>IIII', buf,index)
    index += struct.calcsize('>IIII')
    ims = np.zeros((numRows*numColumns,numImages))
    for i in range(0,numImages,1):
        im = struct.unpack_from('>784B',buf,index)
        index += struct.calcsize('>784B')
        im = np.array(im)
        ims[:,i] = im[:]
        k += 1
    binfile.close()
    ims = ims/255.
    return ims


def ReadMNISTLabels(filename):
    #read labels
    binfile = open(filename,'rb')
    buf = binfile.read()
    index = 0
    magic, numLabels = struct.unpack_from('>II',buf,index)
    labels = np.zeros((numLabels),dtype = int)
    index += struct.calcsize('>II')
    for i in range(0,numLabels,1):
        label = struct.unpack_from('>B',buf,index)
        index += struct.calcsize('>B')
        labels[i] = int(label[0])
    binfile.close()
    return labels


def ReadMNIST():
    #read train images
    trainImagesFile = '../MNIST/train-images-idx3-ubyte'
    trainImages = ReadMNISTImages(trainImagesFile)
    #read train labels
    trainLabelsFile = '../MNIST/train-labels-idx1-ubyte'
    trainLabels = ReadMNISTLabels(trainLabelsFile)

    return [trainImages,trainLabels]


def preData(data, mean, pstd):
    data = data - mean
    data[data > pstd] = pstd
    data[data < -pstd] = -pstd
    data = data/pstd
    data = (data+1)*0.4+0.1
    return data


def GetDivideSets():
    #读取MINST数据集
    Images, Labels = ReadMNIST()
    #取标签<=4 的作为标签数据集
    labeledSet = np.where(Labels <= 4)[0]
    #取标签>4 的作为无标签数据集
    unlabeledSet = np.where(Labels >= 5)[0]

    #因数据集太大,进行自编码会导致内存溢出,因此只选1/3的数据来做实验
    numTrain = labeledSet.shape[0]/3
    trainSet = labeledSet[0:numTrain]
    testSet = labeledSet[numTrain:2*numTrain]

    unlabeledData = Images[:,unlabeledSet]
    unlabeledData = unlabeledData[:, 0:unlabeledData.shape[1]/3]

    trainData = Images[:, trainSet]
    trainLabels = Labels[trainSet]

    testData = Images[:, testSet]
    testLabels = Labels[testSet]
    print "unlabeledData length:"+str(unlabeledData.shape[1])
    print "trainData length:"+str(trainData.shape[1])
    print "testData length:"+str(testData.shape[1])

    return unlabeledData, trainData, trainLabels, testData, testLabels


def feedforward(theta, hidden, visible, data):
    W1 = theta[0:(hidden*visible)].reshape(hidden,visible)
    b1 = theta[(hidden*visible*2):(hidden*visible*2+hidden)].reshape(hidden,1)
    m = data.shape[1]
    Z2 = np.dot(W1, data)+np.repeat(b1,m,1)
    A2 = sparseAutoencoder.sigmoid(Z2)
    return A2


if __name__ == '__main__':
    #step1 参数初始化
    inputSize = 28*28
    numClasses = 5
    hiddenSize = 200
    sparsityParam = 0.1
    la = 3e-3
    beta = 3

    #step2 获取无标签数据集, 有标签训练数据集, 有标签测试数据集
    unlabeledData, trainData, trainLabels, testData, testLabels = GetDivideSets()

    #step3 用无标签数据集训练自编码的特征
    theta = sparseAutoencoder.initiallize(hiddenSize, inputSize)
    [X,cost,d]=sop.fmin_l_bfgs_b(lambda (x) :sparseAutoencoder.sparseAutoencoderCost(x, inputSize, \
    hiddenSize, la, sparsityParam, beta, unlabeledData),x0=theta,maxiter=400,disp=1)
    W1 = X[0:hiddenSize*inputSize].reshape(hiddenSize, inputSize)
    W1 = W1.T
    opttheta = X

    #step4 获得有标签训练集的激活值并用来训练softmax分类器的权值
    trainImages = feedforward(opttheta, hiddenSize, inputSize, trainData)
    thetaSoftmax = softmax.initiallize(numClasses, hiddenSize)
    la = 1e-4
    [X,cost,d] = sop.fmin_l_bfgs_b(lambda (x) :softmax.SoftmaxCost(trainImages, trainLabels,x,la,hiddenSize,numClasses),x0=thetaSoftmax,maxiter=100,disp=1)

    #step5 获得有标签测试集的激活值并且给出softmax分类准确率, 5分类的准确率应该在98%附近
    testImages = feedforward(opttheta, hiddenSize, inputSize,          testData)
    optthetaSoftmax = X
    accuracy = softmax.predict(optthetaSoftmax,testImages,testLabels,hiddenSize,numClasses)
    print accuracy

    #画出权值图像
    sparseAutoencoder.showimg(W1, hiddenSize, 28)
