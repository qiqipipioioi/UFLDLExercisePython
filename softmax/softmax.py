#!/usr/bin/python
#coding:utf-8
__author__ = "XunTu qiqipipioioi@qq.com"

import numpy as np
import scipy.optimize as sop
import struct
import matplotlib.pyplot as plt
import random


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


def ReadMINST():
    #read train images
    trainImagesFile = '../MNIST/train-images-idx3-ubyte'
    trainImages = ReadMNISTImages(trainImagesFile)
    #read train labels
    trainLabelsFile = '../MNIST/train-labels-idx1-ubyte'
    trainLabels = ReadMNISTLabels(trainLabelsFile)
    #read test images
    testImagesFile = '../MNIST/t10k-images-idx3-ubyte'
    testImages = ReadMNISTImages(testImagesFile)
    #read test labels
    testLabelsFile = '../MNIST/t10k-labels-idx1-ubyte'
    testLabels = ReadMNISTLabels(testLabelsFile)

    return [trainImages,trainLabels,testImages,testLabels]


def computeNumericalGradient(J, theta):
    numgrad = np.zeros(theta.shape)
    epsilon = 1e-4
    n = theta.shape[0]
    E = np.eye(n)
    for i in range(0,n,1):
        print i
        delta = E[:,i]*epsilon
        numgrad[i] = (J(theta+delta)[0]-J(theta-delta)[0])/(epsilon*2.)
    return numgrad


def initiallize(numClasses,inputSize):
    theta = 0.005*np.random.random((numClasses,inputSize))
    theta = theta.ravel()
    return theta


def SoftmaxCost(data,label,theta,la,inputSize,numClasses):
    theta = theta.reshape(numClasses,inputSize)
    counts = data.shape[1]
    Mat = np.dot(theta, data)
    Mat = Mat-np.max(Mat,0).reshape(1,counts)
    Mat = np.exp(Mat)
    probMat = Mat/np.sum(Mat,0).reshape(1,counts)
    LabelMat = np.zeros((numClasses,counts))
    for i in range(0,counts,1):
        LabelMat[label[i],i] = 1
    cost = -1/float(counts)*np.sum(LabelMat*np.log(probMat))+la/2*np.sum(theta*theta)
    grad = -1/float(counts)*np.dot(LabelMat-probMat,data.T)+la*theta
    grad = grad.ravel()
    return [cost,grad]


def predict(theta,testImages,testLabels,inputSize,numClasses):
    theta = theta.reshape(numClasses,inputSize)
    counts = testImages.shape[1]
    Mat = np.dot(theta, testImages)
    predict = Mat.argmax(0)
    testLabels = np.array(testLabels)
    print predict
    print testLabels
    accuracy = np.mean(predict == testLabels)
    return accuracy


def showimgs(Images,Labels):
    length = Images.shape[1]
    samples = []
    num1 = int(random.uniform(0,length))
    num2 = int(random.uniform(0,length))
    num3 = int(random.uniform(0,length))
    samples.append(Images[:,num1].reshape(28,28))
    samples.append(Images[:,num2].reshape(28,28))
    samples.append(Images[:,num3].reshape(28,28))

    plt.subplot(131)
    plt.imshow(samples[0],cmap='gray')
    plt.title(Labels[num1])
    plt.subplot(132)
    plt.imshow(samples[1],cmap='gray')
    plt.title(Labels[num2])
    plt.subplot(133)
    plt.imshow(samples[2],cmap='gray')
    plt.title(Labels[num3])
    plt.show()


if __name__ == '__main__':
    #step1 读取MNIST手写数据集
    [trainImages,trainLabels,testImages,testLabels] = ReadMINST()

    #参数设置
    la = 1e-4
    inputSize = 28*28
    numClasses = 10
    theta = initiallize(numClasses,inputSize)

    #随机选取3个手写数据并检测标签是否正确
    showimgs(trainImages,trainLabels)

    #step2 梯度检测
    [cost,grad] = SoftmaxCost(trainImages,trainLabels,theta,la,inputSize,numClasses)
    numgrad = computeNumericalGradient(lambda (x):SoftmaxCost(trainImages,trainLabels,x,la,inputSize,numClasses),theta)
    diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)
    print diff

    #softmax分类, 最后的准确率应该在92%左右
    [X,cost,d] = sop.fmin_l_bfgs_b(lambda (x) :SoftmaxCost(trainImages,trainLabels,x,la,inputSize,numClasses),x0=theta,maxiter=100,disp=1)
    accuracy = predict(X,testImages,testLabels,inputSize,numClasses)
    print accuracy
