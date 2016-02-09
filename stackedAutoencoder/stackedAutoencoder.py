#!/usr/bin/python
#coding:utf-8
__author__ = "XunTu qiqipipioioi@qq.com"

import numpy as np
import scipy.optimize as sop
import scipy.sparse as ssp
import struct
import matplotlib.pyplot as plt
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
    #read test images
    testImagesFile = '../MNIST/t10k-images-idx3-ubyte'
    testImages = ReadMNISTImages(testImagesFile)
    #read test labels
    testLabelsFile = '../MNIST/t10k-labels-idx1-ubyte'
    testLabels = ReadMNISTLabels(testLabelsFile)

    return [trainImages,trainLabels,testImages,testLabels]


def feedforward(theta, hidden, visible, data):
    W1 = theta[0:(hidden*visible)].reshape(hidden,visible)
    b1 = theta[(hidden*visible*2):(hidden*visible*2+hidden)].reshape(hidden,1)
    m = data.shape[1]
    Z2 = np.dot(W1, data)+np.repeat(b1,m,1)
    A2 = sparseAutoencoder.sigmoid(Z2)
    return A2


def stack2param(stack):
    #把权值字典展平,并且返回深度神经网络参数
    params = np.array([])
    for d in range(0,len(stack),1):
        params = np.append(params, stack[d]['w'].ravel())
        params = np.append(params, stack[d]['b'].ravel())
    netconfig = {}
    netconfig['inputsize'] = stack[0]['w'].shape[1]
    netconfig['layersizes'] = []
    for d in range(0,len(stack),1):
        netconfig['layersizes'].append(stack[d]['w'].shape[0])
    return params, netconfig


def param2stack(params, netconfig):
    #从展平的结国和参数返回权值字典
    depth = len(netconfig['layersizes'])
    prevLayerSize = netconfig['inputsize']
    curPos = 0
    stack = {}

    for d in range(0, depth, 1):
        stack[d] = {}
        wlen = netconfig['layersizes'][d]*prevLayerSize
        stack[d]['w'] = params[curPos:curPos+wlen].reshape(netconfig['layersizes'][d], prevLayerSize)
        curPos = curPos + wlen
        blen = netconfig['layersizes'][d]
        stack[d]['b'] = params[curPos:curPos+blen].reshape(netconfig['layersizes'][d], 1)
        curPos = curPos + blen
        prevLayerSize = netconfig['layersizes'][d]

    return stack


def initaillizeAETheta(sae1Theta, sae2Theta, softmaxTheta,hiddenSizeL1, hiddenSizeL2, inputSize):
    stack = {}
    stack[0] = {}
    stack[1] = {}
    stack[0]['w'] = sae1Theta[0:hiddenSizeL1*inputSize].reshape(hiddenSizeL1, inputSize)
    stack[0]['b'] = sae1Theta[2*hiddenSizeL1*inputSize:2*hiddenSizeL1*inputSize+hiddenSizeL1]
    stack[1]['w'] = sae2Theta[0:hiddenSizeL1*hiddenSizeL2].reshape(hiddenSizeL1, hiddenSizeL2)
    stack[1]['b'] = sae2Theta[2*hiddenSizeL2*hiddenSizeL1:2*hiddenSizeL2*hiddenSizeL1+hiddenSizeL2]
    params, netconfig = stack2param(stack)
    stackedAETheta = np.append(softmaxTheta.ravel(), params)

    return stackedAETheta, netconfig


def stackedAECost(theta, inputSize, hiddenSize, numClasses, netconfig, la, data, labels):
    #获得权值字典
    softmaxTheta = theta[0 : hiddenSize*numClasses].reshape(numClasses, hiddenSize)
    stack = param2stack(theta[hiddenSize*numClasses :], netconfig)

    #权值梯度初始化
    softmaxThetaGrad = np.zeros(softmaxTheta.shape)
    stackgrad = {}
    for d in range(0, len(stack), 1):
        stackgrad[d] = {}
        stackgrad[d]['w'] = np.zeros(stack[d]['w'].shape)
        stackgrad[d]['b'] = np.zeros(stack[d]['b'].shape)

    #这里用稀疏矩阵的方式得到标签矩阵,比softmax实验里的方法看起来要更合理
    M = data.shape[1]
    cols = np.arange(0, len(labels))
    rows = np.array(labels)
    sparseData = np.zeros((len(labels),)) + 1
    groundTruth = ssp.coo_matrix((sparseData, (rows, cols)), shape = (numClasses, len(labels))).todense()
    groundTruth = np.array(groundTruth)

    #前向传播神经网络
    depth = len(stack)
    Z = []
    Z.append(np.array([]))
    A = []
    A.append(data)
    for d in range(0, depth, 1):
        z = np.dot(stack[d]['w'], A[d]) + np.repeat(stack[d]['b'], M, 1)
        Z.append(z)
        a = sigmoid(z)
        A.append(a)

    #得到softmax的概率矩阵
    counts = M
    Mat = np.dot(softmaxTheta, A[depth])
    Mat = Mat - np.max(Mat, 0).reshape(1, counts)
    Mat = np.exp(Mat)
    probMat = Mat/np.sum(Mat, 0).reshape(1, counts)

    #计算损失函数和softmax损失梯度
    cost = -1/float(counts)*np.sum(groundTruth*np.log(probMat)) + la/2*np.sum(softmaxTheta*softmaxTheta)
    softmaxgrad = -1/float(counts)*np.dot(groundTruth-probMat, A[depth].T) + la*softmaxTheta

    #计算每一层神经网络的权值梯度
    delta = {}
    delta[depth - 1] = -np.dot(softmaxTheta.T, groundTruth - probMat) * dsigmoid(Z[depth])
    for d in range(depth-2, -1, -1):
        delta[d] = np.dot(stack[d+1]['w'].T, delta[d+1])*dsigmoid(Z[d+1])

    for d in range(depth-1, -1, -1):
        stackgrad[d]['w'] = (1/float(M))*np.dot(delta[d], A[d].T)
        stackgrad[d]['b'] = (1/float(M))*np.sum(delta[d], 1)
        stackgrad[d]['b'] = stackgrad[d]['b'].reshape(stackgrad[d]['b'].shape[0], 1)

    #把权值展平
    grad = np.append(softmaxgrad.ravel(), stack2param(stackgrad)[0])

    return cost, grad


def predict(theta, inputSize, hiddenSize, numClasses, netconfig, testData, testLabels):
    #预测函数
    softmaxTheta = theta[0 : hiddenSize*numClasses].reshape(numClasses, hiddenSize)
    stack = param2stack(theta[hiddenSize*numClasses :], netconfig)

    M = testData.shape[1]
    depth = len(stack)
    Z = []
    Z.append(np.array([]))
    A = []
    A.append(testData)
    for d in range(0, depth, 1):
        z = np.dot(stack[d]['w'], A[d]) + np.repeat(stack[d]['b'], M, 1)
        Z.append(z)
        a = sigmoid(z)
        A.append(a)

    counts = M
    Mat = np.dot(softmaxTheta, A[depth])
    predict = Mat.argmax(0)
    testLabels = np.array(testLabels)
    accuracy = np.mean(predict == testLabels)

    return accuracy


def sigmoid(x):
    return 1/(1+np.exp(-x))


def dsigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))


def computeNumericalGradient(J, theta):
    numgrad = np.zeros(theta.shape)
    epsilon = 1e-4
    n = theta.shape[0]
    E = np.eye(n)
    for i in range(0,n,1):
        print i
        delta = E[:,i]*epsilon
        numgrad[i] = (J(theta+delta)[0]-J(theta-delta)[0])/(epsilon*2.0)
    return numgrad


def checkStackedAECost():
    #制造一个小的数据集来检测栈式神经网络损失函数是否正确
    inputSize = 4
    hiddenSize = 5
    la = 0.01
    data = np.random.random((inputSize, 5))
    labels = [0,1,1,1,0]
    numClasses = 2

    stack = {}
    stack[0] = {}
    stack[0]['w'] = 0.1*np.random.random((3, inputSize))
    stack[0]['b'] = np.zeros((3, 1))
    stack[1] = {}
    stack[1]['w'] = 0.1*np.random.random((hiddenSize, 3))
    stack[1]['b'] = np.zeros((hiddenSize, 1))
    softmaxAETheta = 0.005 * np.random.random((hiddenSize * numClasses, 1))
    stackparams, netconfig = stack2param(stack)
    stackedAETheta = np.append(softmaxAETheta.ravel(), stackparams)

    cost, grad = stackedAECost(stackedAETheta, inputSize, hiddenSize, numClasses, netconfig, la, data, labels)

    numgrad = computeNumericalGradient(lambda (x) : stackedAECost(x, inputSize, hiddenSize, numClasses, netconfig, la, data, labels), stackedAETheta)

    diff = np.linalg.norm(numgrad - grad)/np.linalg.norm(numgrad + grad)
    print diff


if __name__ == '__main__':
    #step1 两层隐含层的栈式神经网络的参数设置
    inputSize = 28*28
    numClasses = 10
    hiddenSizeL1 = 200
    hiddenSizeL2 = 200
    sparsityParam = 0.1
    la = 3e-3
    beta = 3

    #step2 读取MNIST数据集,只取1/3的数据集防止内存溢出
    trainData, trainLabels, testData, testLabels = ReadMNIST()
    length = trainData.shape[1]/3
    trainData = trainData[:, 0:length]
    trainLabels = trainLabels[0:length]
    sae1Theta = sparseAutoencoder.initiallize(hiddenSizeL1, inputSize)

    #step3 训练第一层神经网络的权值
    [sae1OptTheta, cost, d]=sop.fmin_l_bfgs_b(lambda (x) :sparseAutoencoder.sparseAutoencoderCost(x, inputSize, \
    hiddenSizeL1, la, sparsityParam, beta, trainData),x0=sae1Theta,maxiter=400,disp=1)
    sae1Features = feedforward(sae1OptTheta, hiddenSizeL1, inputSize, trainData)

    #step4 训练第二层神经网络的权值
    sae2Theta = sparseAutoencoder.initiallize(hiddenSizeL2, hiddenSizeL1)
    [sae2OptTheta, cost, d]=sop.fmin_l_bfgs_b(lambda (x) : sparseAutoencoder.sparseAutoencoderCost(x, hiddenSizeL1, \
    hiddenSizeL2, la, sparsityParam, beta, sae1Features),x0=sae2Theta, maxiter=400,disp=1)
    sae2Features = feedforward(sae2OptTheta, hiddenSizeL2, hiddenSizeL1, sae1Features)

    #step5 训练softmax权值
    saeSoftmaxTheta = softmax.initiallize(numClasses, hiddenSizeL2)
    laSoftmax = 1e-4
    [saeSoftmaxOptTheta, cost, d] = sop.fmin_l_bfgs_b(lambda (x) :softmax.SoftmaxCost(sae2Features, \
    trainLabels, x, laSoftmax, hiddenSizeL2, numClasses), x0=saeSoftmaxTheta, maxiter=100, disp=1)

    #step6 微调栈式神经网络的所有权值
    stackedAETheta, netconfig = initaillizeAETheta(sae1OptTheta, sae2OptTheta, saeSoftmaxOptTheta, hiddenSizeL1, hiddenSizeL2, inputSize)
    [stackedAEOptTheta, cost, d] = sop.fmin_l_bfgs_b(lambda (x):stackedAECost(x, inputSize, hiddenSizeL2, \
    numClasses, netconfig, la, trainData, trainLabels), x0 = stackedAETheta, maxiter = 400, disp=1)

    #step7 给出栈式神经网络的10分类的准确率, 微调前的应该在90%左右, 微调后在97%左右
    accuracyBeforeFine = predict(stackedAETheta, inputSize, hiddenSizeL2, numClasses, netconfig, testData, testLabels)
    accuracyAfterFine = predict(stackedAEOptTheta, inputSize, hiddenSizeL2, numClasses, netconfig, testData, testLabels)
    print 'accuracyBeforeFine: '+str(accuracyBeforeFine)
    print 'accuracyAfterFine: '+str(accuracyAfterFine)

    #step8 第一层,第二层神经网络权值图像的展示
    stack = param2stack(stackedAEOptTheta[hiddenSizeL1*numClasses :], netconfig)
    W1 = stack[0]['w']
    W2 = stack[1]['w']
    W12 = np.dot(W1.T, W2)
    W1 = W1.T
    sparseAutoencoder.showimg(W1, hiddenSizeL1, 28)
    sparseAutoencoder.showimg(W12, hiddenSizeL1, 28)
