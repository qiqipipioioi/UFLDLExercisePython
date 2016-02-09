#!/usr/bin/python
#coding:utf-8
__author__ = "XunTu qiqipipioioi@qq.com"

import numpy as np
from matplotlib import pyplot as plt
import scipy.optimize as sop
import matplotlib as mpl
import cv2
import scipy.io as sio
import scipy.signal as sig
import random
import math
import sys
sys.path.append('../softmax')
import softmax


def showimg(img, n, sz):
    length = img.shape[1]
    img = img[:,np.random.choice(length,n,replace=False)]
    length = img.shape[1]
    lst = []
    for j in range(0,length):
        img1 = img[:,j]
        img2 = img1.reshape(sz,sz)
        lst.append(img2)
    la = np.zeros((sz,2))
    wide = int(np.sqrt(length))
    height = length/wide
    for i in range(0,height,1):
        for j in range(0,wide,1):
            if j == 0:
                img_m = lst[height*i]
            else:
                img_m = np.append(img_m,la,1)
                img_m = np.append(img_m,lst[height*i+j],1)
        ma = np.zeros((2,img_m.shape[1]))
        if i == 0:
            img_out = img_m
        else:
            img_out = np.append(img_out,ma,0)
            img_out = np.append(img_out,img_m,0)
    return img_out


def showColorimg(img, n, sz):
    sz2 = sz*sz
    A = img[0:sz2, :]
    B = img[sz2:2*sz2, :]
    C = img[2*sz2: , :]

    sed = int(random.random()*100)
    np.random.seed(sed)
    A_out = showimg(A, n, sz)
    np.random.seed(sed)
    B_out = showimg(B, n, sz)
    np.random.seed(sed)
    C_out = showimg(C, n ,sz)
    A_out = A_out.reshape(A_out.shape+(1,))
    print A_out.shape
    B_out = B_out.reshape(B_out.shape+(1,))
    print B_out.shape
    C_out = C_out.reshape(C_out.shape+(1,))
    print C_out.shape
    img_out = np.concatenate((A_out, B_out, C_out), 2)
    plt.imshow(img_out)
    plt.show()


def imgTran(img, sz):
    #原始图像展平,不能直接用reshape,中间要用到一次转置
    length = img.shape[3]
    img1 = img.reshape(sz*sz, 3, length)
    img2 = np.zeros((sz*sz*3, length))
    for i in range(0, length):
        img2[:, i] = img1[:, :, i].T.reshape(sz*sz*3)
    return img2


def initiallize(hidden,visible):
    r = np.sqrt(6)/np.sqrt(hidden+visible+1)
    W1 = np.random.random((hidden, visible))*2*r-r
    W2 = np.random.random((visible, hidden))*2*r-r
    b1 = np.zeros((hidden, 1))
    b2 = np.zeros((visible, 1))
    theta = np.append(W1.ravel(),W2.ravel())
    theta = np.append(theta,b1.ravel())
    theta = np.append(theta,b2.ravel())
    return theta


def cnnConvolve(patchDim, numFeatures, images, W, b, white, meanPatch):
    #计算各个图像的卷积特征
    patchSize = patchDim * patchDim
    numImages = images.shape[3]
    imageDim = images.shape[0]
    imageChannels = images.shape[2]

    WT = np.dot(W, white)
    b = b.reshape(b.shape[0], 1)
    b_mean = b - np.dot(WT, meanPatch)
    convolvedFeatures = np.zeros((numFeatures, numImages, imageDim - patchDim + 1, imageDim - patchDim + 1))
    #三层循环,对每一个卷积核,每一个图像,得到卷积特征
    for imageNum in range(0, numImages):
        print imageNum
        for featureNum in range(0, numFeatures):
            convolvedImage = np.zeros((imageDim - patchDim + 1, imageDim - patchDim + 1))
            for channel in range(0, imageChannels):
                offset = channel * patchSize
                feature = WT[featureNum,offset:offset+patchSize].reshape(patchDim, patchDim)
                im = images[:, :, channel, imageNum]
                #这里跟matlab一样,convolve2d函数会先把卷积核翻转过来,因为实验并不需要做翻转,因此要先翻转过来
                flpfeature = np.fliplr(np.flipud(feature))
                convolvedChannel = sig.convolve2d(im, flpfeature, 'valid')
                convolvedImage = convolvedImage + convolvedChannel
            convolvedImage = sigmoid(convolvedImage + b_mean[featureNum, 0])
            convolvedFeatures[featureNum, imageNum, :, :] = convolvedImage
    return convolvedFeatures


def cnnPool(poolDim, convolvedFeatures):
    #池化,简单用平均值的方法进行池化
    numImages = convolvedFeatures.shape[1]
    numFeatures = convolvedFeatures.shape[0]
    convolvedDim = convolvedFeatures.shape[2]
    resultDim = int(float(convolvedDim)/poolDim)
    pooledFeatures = np.zeros((numFeatures, numImages, resultDim, resultDim))

    for imageNum in range(0, numImages):
        for featureNum in range(0, numFeatures):
            for poolRow in range(0, resultDim):
                offsetRow = poolRow * poolDim
                for poolCol in range(0, resultDim):
                    offsetCol = poolCol * poolDim
                    patch = convolvedFeatures[featureNum, imageNum, offsetRow:offsetRow + poolDim, offsetCol:offsetCol + poolDim]
                    pooledFeatures[featureNum, imageNum, poolRow, poolCol] = np.mean(patch)

    return pooledFeatures


def feedForward(theta, hidden, visible, data):
    W1 = theta[0, 0:(hidden*visible)].reshape(hidden,visible)
    b1 = theta[0, (hidden*visible*2):(hidden*visible*2+hidden)].reshape(hidden,1)
    m = data.shape[1]
    Z2 = np.dot(W1, data)+np.repeat(b1,m,1)
    A2 = sigmoid(Z2)
    return A2


def sigmoid(x):
    return 1/(1+np.exp(-x))


def dsigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))


if __name__ == '__main__':
    #step1 读取训练数据集和测试数据集,并随即展示训练图像
    Mat = sio.loadmat('stlTrainSubset.mat')
    numTrainImages = Mat['numTrainImages']
    trainImages = Mat['trainImages']
    trainLabels = Mat['trainLabels']
    trainLabels = np.squeeze(trainLabels - 1)
    trainImagesFlat = imgTran(trainImages, 64)
    showColorimg(trainImagesFlat, 16, 64)

    testMat = sio.loadmat('stlTestSubset.mat')
    numTestImages = testMat['numTestImages']
    testImages = testMat['testImages']
    testLabels = testMat['testLabels']
    testLabels = np.squeeze(testLabels - 1)
    testImagesFlat = imgTran(testImages, 64)

    #step2 卷积特征提取的参数设置
    imageDim = 64
    imageChannels = 3
    patchDim = 8
    numPatches = 50000
    visibleSize = patchDim * patchDim * imageChannels
    outputSize = visibleSize
    hiddenSize = 400
    epsilon = 0.1
    poolDim = 19

    #step3 读取上一个实验得到的特征,ZCAWhite,meanPatch
    resultMat = sio.loadmat('resultFeatures.mat')
    optTheta = resultMat['optTheta']
    W = optTheta[0, 0: visibleSize * hiddenSize].reshape(hiddenSize, visibleSize)
    b = optTheta[0, 2 * hiddenSize * visibleSize : 2 * hiddenSize * visibleSize + hiddenSize]
    white = resultMat['white']
    meanPatch = resultMat['meanPatch']
    meanPatch = meanPatch.reshape(meanPatch.shape[0], 1)
    showColorimg((np.dot(W, white).T + 1) / 2., 400, 8)

    #step4 测试卷积特征提取
    convImages = trainImages[:, :, :, 0:8]
    convolvedFeatures = cnnConvolve(patchDim, hiddenSize, convImages, W, b, white, meanPatch)

    for i in range(0, 1000):
        featureNum = int(random.uniform(0, hiddenSize))
        imageNum = int(random.uniform(0, 8))
        imageRow = int(random.uniform(0, imageDim - patchDim + 1))
        imageCol = int(random.uniform(0, imageDim - patchDim + 1))
        patch = convImages[imageRow:imageRow + patchDim, imageCol:imageCol + patchDim, :, imageNum]
        patch = patch.reshape(patchDim, patchDim, imageChannels, 1)
        patch = imgTran(patch, patchDim)
        patch = patch - meanPatch
        patch = np.dot(white, patch)

        features = feedForward(optTheta, hiddenSize, visibleSize, patch)
        if abs(features[featureNum, 0] - convolvedFeatures[featureNum, imageNum, imageRow, imageCol]) > 1e-9:
            print 'fail'
    print 'Convolve test passed!'

    #step5 测试池化正确与否
    pooledFeatures = cnnPool(poolDim, convolvedFeatures)
    testMatrix = np.arange(0, 64).reshape(8, 8)
    expectedMatrix = np.array([[np.mean(testMatrix[0:4, 0:4]), np.mean(testMatrix[0:4, 4:8])], [np.mean(testMatrix[4:8, 0:4]), np.mean(testMatrix[4:8, 4:8])]])
    testMatrix = testMatrix.reshape(1, 1, 8, 8)
    pooledFeatures = np.squeeze(cnnPool(4, testMatrix))
    if (pooledFeatures == expectedMatrix).all():
        print 'Pool test passed!'

    #step6 对数据集进行卷积特征提取,一次使用50个卷积核,这个步骤用了7个小时,CPU i3,4G内存, 所以结束了一定要把结果存储下来
    stepSize = 50
    pooledFeaturesTrain = np.zeros((hiddenSize, numTrainImages, int((imageDim - patchDim + 1)/poolDim), int((imageDim - patchDim + 1)/poolDim)))
    pooledFeaturesTest = np.zeros((hiddenSize, numTestImages, int((imageDim - patchDim + 1)/poolDim), int((imageDim - patchDim + 1)/  poolDim)))
    for convPart in range(0, hiddenSize/stepSize):
        print 'convPart: ' + str(convPart)
        featureStart = convPart * stepSize
        featureEnd = (convPart + 1) * stepSize
        Wt = W[featureStart:featureEnd, :]
        bt = b[featureStart:featureEnd]
        convolvedFeaturesThis = cnnConvolve(patchDim, stepSize, trainImages, Wt, bt, white, meanPatch)
        pooledFeaturesThis = cnnPool(poolDim, convolvedFeaturesThis)
        pooledFeaturesTrain[featureStart:featureEnd, :, :, :] = pooledFeaturesThis
        convolvedFeaturesThis = cnnConvolve(patchDim, stepSize, testImages, Wt, bt, white, meanPatch)
        pooledFeaturesThis = cnnPool(poolDim, convolvedFeaturesThis)
        pooledFeaturesTest[featureStart:featureEnd, :, :, :] = pooledFeaturesThis

    cnnPooledFeatures = {}
    cnnPooledFeatures['pooledFeaturesTrain'] = pooledFeaturesTrain
    cnnPooledFeatures['pooledFeaturesTest'] = pooledFeaturesTest
    sio.savemat('cnnPooledFeatures.mat', cnnPooledFeatures)

    #step7 用卷积特征来进行softmax预测,这个就很快了,十几秒, 正确率为80%左右
    Mat1 = sio.loadmat('cnnPooledFeatures.mat')
    pooledFeaturesTrain = Mat1['pooledFeaturesTrain']
    pooledFeaturesTest = Mat1['pooledFeaturesTest']
    softmaxLambda = 1e-4
    numClasses = 4
    inputSize = int(pooledFeaturesTrain.size/numTrainImages)
    print inputSize
    softmaxX = np.transpose(pooledFeaturesTrain, (0, 2, 3, 1))
    softmaxX = softmaxX.reshape(inputSize, numTrainImages)
    softmaxY = trainLabels
    theta = softmax.initiallize(numClasses, inputSize)
    [X,cost,d] = sop.fmin_l_bfgs_b(lambda (x) :softmax.SoftmaxCost(softmaxX, softmaxY, x, softmaxLambda, inputSize, numClasses),x0=theta,maxiter=200,disp=1)

    softmaxX = np.transpose(pooledFeaturesTest, (0, 2, 3, 1))
    softmaxX = softmaxX.reshape(inputSize, numTestImages)
    softmaxY = testLabels
    accuracy = softmax.predict(X,softmaxX,softmaxY,inputSize,numClasses)
    print accuracy

