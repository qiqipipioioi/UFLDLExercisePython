#!/usr/bin/python
#coding:utf-8
__author__ = "XunTu qiqipipioioi@qq.com"

import numpy as np
from matplotlib import pyplot as plt
import scipy.optimize as sop
import matplotlib as mpl
import cv2
import scipy.io as sio
import random
import math


def data_prepare():
    Mat = sio.loadmat('example/IMAGES.mat')['IMAGES']
    patchsize = 8
    numpatches = 10000
    patches = np.zeros((patchsize*patchsize, numpatches))
    for imageNum in range(0,10,1):
        rowNum = Mat[:,:,imageNum].shape[0]
        colNum = Mat[:,:,imageNum].shape[1]
        for patchNum in range(0,1000,1):
            xPos = int(random.uniform(0,rowNum-patchsize))
            yPos = int(random.uniform(0,colNum-patchsize))
            patches[:,imageNum*1000+patchNum] = Mat[xPos:(xPos+8),yPos:(yPos+8),imageNum].reshape(64)
    patches = patches - np.mean(patches)
    pstd = 3*np.std(patches)
    patches[patches > pstd] = pstd
    patches[patches < -pstd] = -pstd
    patches = patches/pstd
    patches = (patches+1)*0.4+0.1
    return patches


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
    #和之前不同的是,这里的图像不再是灰度图,而是三通道的彩色图,因此用这个函数来展示图像
    A = img[0:64, :]
    B = img[64:128, :]
    C = img[128: , :]

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


def ZCAwhite(patches):
    #对数据集进行ZCAWhite预处理
    epsilon = 0.1
    meanPatch = np.mean(patches, 1).reshape(patches.shape[0], 1)
    patches = patches - meanPatch
    sigma = np.cov(patches)
    laValue, laMat = np.linalg.eig(sigma)
    rot = np.dot(laMat.T, patches)
    for i in range(0,len(laValue),1):
        rot[i,:] = rot[i,:]/(np.sqrt(laValue[i]+0.1))
    ZCAW = np.dot(laMat, rot)

    patches = np.matrix(patches)
    ZCAW = np.matrix(ZCAW)
    white = ZCAW * patches.I
    white = np.array(white)
    ZCAW = np.array(ZCAW)
    return ZCAW, white, meanPatch


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


def linearAutoencoderCost(theta, visible, hidden, la, sparsityParam, beta, data):
    #线性自编码和之前自编码的不同在于输出函数用y=x,而不再用sigmoid函数,可以不再对输入值有[-1,1]的要求
    W1 = theta[0:(hidden*visible)].reshape(hidden,visible)
    W2 = theta[(hidden*visible):(visible*hidden*2)].reshape(visible,hidden)
    b1 = theta[(hidden*visible*2):(hidden*visible*2+hidden)].reshape(hidden,1)
    b2 = theta[(hidden*visible*2+hidden):].reshape(visible,1)
    cost = 0
    W1grad = np.zeros((W1.shape))
    W2grad = np.zeros((W2.shape))
    b1grad = np.zeros((b1.shape))
    b2grad = np.zeros((b2.shape))

    Jcost = 0
    Jweight = 0
    Jsparse = 0
    n = data.shape[0]
    m = data.shape[1]

    Z2 = np.dot(W1, data)+np.repeat(b1,m,1)
    A2 = sigmoid(Z2)
    Z3 = np.dot(W2, A2)+np.repeat(b2,m,1)
    #仅仅在这里有所不同
    A3 = Z3

    Jcost = (0.5/m)*np.sum((A3-data)*(A3-data))
    Jweight = 0.5*(np.sum(W1*W1))+0.5*(np.sum(W2*W2))
    rho = (1/float(m))*np.sum(A2,1)
    rho = rho.reshape(len(rho),1)
    Jsparse = np.sum(sparsityParam*np.log(sparsityParam/rho))+ \
            np.sum((1-sparsityParam)*np.log((1-sparsityParam)/(1-rho)))
    cost = Jcost + la*Jweight + beta*Jsparse

    D3 = -(data-A3)
    sterm = beta*(-sparsityParam/rho+(1-sparsityParam)/(1-rho))
    D2 = (np.dot(W2.T,D3)+np.repeat(sterm,m,1))*dsigmoid(Z2)

    W1grad = W1grad+np.dot(D2,data.T)
    W1grad = (1/float(m))*W1grad+la*W1

    W2grad = W2grad+np.dot(D3,A2.T)
    W2grad = (1/float(m))*W2grad+la*W2

    b1grad = b1grad+np.sum(D2,1).reshape(hidden,1)
    b1grad = (1/float(m))*b1grad

    b2grad = b2grad+np.sum(D3,1).reshape(visible,1)
    b2grad = (1/float(m))*b2grad

    grad = np.append(W1grad.ravel(),W2grad.ravel())
    grad = np.append(grad,b1grad.ravel())
    grad = np.append(grad,b2grad.ravel())

    return [cost,grad]


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
        delta = E[:,i]*epsilon
        numgrad[i] = (J(theta+delta)[0]-J(theta-delta)[0])/(epsilon*2.0)
    return numgrad


def checkNumericalGradient():
    beta = 3
    hiddenSize = 5
    visibleSize = 8
    la = 3e-3
    sparsityParam = 0.035
    patches = np.random.random((8, 10))
    theta = initiallize(hiddenSize, visibleSize)
    cost, grad = linearAutoencoderCost(theta, visibleSize, hiddenSize, la, sparsityParam, beta, patches)
    numgrad = computeNumericalGradient(lambda (x): linearAutoencoderCost(x, visibleSize, hiddenSize,  la, sparsityParam, beta, patches), theta)
    diff = np.linalg.norm(grad - numgrad)/np.linalg.norm(grad + numgrad)
    print diff


if __name__ == '__main__':
    #step1 初始化参数
    imageChannels = 3
    patchDim = 8
    numPatches = 100000
    visibleSize = patchDim * patchDim * imageChannels
    outputSize = visibleSize
    hiddenSize = 400
    la = 3e-3
    sparsityParam = 0.035
    beta = 5
    theta = initiallize(hiddenSize, visibleSize)

    #step2 读取样本数据
    mat = sio.loadmat('stlSampledPatches.mat')
    Mat = mat['patches']
    ZCAWhiteMat, white, meanPatch = ZCAwhite(Mat)
    showColorimg(Mat, 100, 8)

    #step3 训练线性编码器
    [X,cost,d]=sop.fmin_l_bfgs_b(lambda (x) :linearAutoencoderCost(x, visibleSize, \
    hiddenSize, la, sparsityParam, beta, ZCAWhiteMat), x0=theta, maxiter=400, disp=1)

    #step4 权值展现并把权值,ZCAWhite,meanPatch存储下来供下一个实验使用
    W1 = X[0:hiddenSize*visibleSize].reshape(hiddenSize,visibleSize)
    img = np.dot(W1, white).T
    img = img + 1
    img = img / 2.
    resultMat = {}
    resultMat['optTheta'] = X
    resultMat['white'] = white
    resultMat['meanPatch'] = meanPatch
    sio.savemat('resultFeatures.mat', resultMat)
    showColorimg(img, 400, 8)
