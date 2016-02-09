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
    #数据准备,由于sigmoid函数的取值范围为(-1,1),所以需要把输入数据归一到(-1,1)的区间内,这样才能完成自编码.95%的数据落在3sigma范围内,故作截断处理
    Mat = sio.loadmat('IMAGES.mat')['IMAGES']
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
    #从图像集合中随即选取n个图像,然后把这n个图像拼凑成一张图,便于展示
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
    print img_out.shape
    cmap = mpl.cm.gray_r
    norm = mpl.colors.Normalize(vmin=0)
    plt.imshow(img_out,cmap=cmap)
    plt.show()


def initiallize(hidden, visible):
    #初始化参数
    r = np.sqrt(6)/np.sqrt(hidden+visible+1)
    W1 = np.random.random((hidden, visible))*2*r-r
    W2 = np.random.random((visible, hidden))*2*r-r
    b1 = np.zeros((hidden, 1))
    b2 = np.zeros((visible, 1))
    theta = np.append(W1.ravel(),W2.ravel())
    theta = np.append(theta,b1.ravel())
    theta = np.append(theta,b2.ravel())
    return theta


def sparseAutoencoderCost(theta, visible, hidden, la, sparsityParam, beta, data):
    #把展平的参数重新变形为需要的参数形状
    W1 = theta[0:(hidden*visible)].reshape(hidden,visible)
    W2 = theta[(hidden*visible):(visible*hidden*2)].reshape(visible,hidden)
    b1 = theta[(hidden*visible*2):(hidden*visible*2+hidden)].reshape(hidden,1)
    b2 = theta[(hidden*visible*2+hidden):].reshape(visible,1)
    W1grad = np.zeros((W1.shape))
    W2grad = np.zeros((W2.shape))
    b1grad = np.zeros((b1.shape))
    b2grad = np.zeros((b2.shape))

    n = data.shape[0]
    m = data.shape[1]

    #正向传播神经网络
    Z2 = np.dot(W1, data)+np.repeat(b1,m,1)
    A2 = sigmoid(Z2)
    Z3 = np.dot(W2, A2)+np.repeat(b2,m,1)
    A3 = sigmoid(Z3)

    #分别计算直接误差,权值惩罚和稀疏性惩罚,最后再加和起来
    Jcost = (0.5/m)*np.sum((A3-data)*(A3-data))
    Jweight = 0.5*(np.sum(W1*W1))+0.5*(np.sum(W2*W2))
    rho = (1/float(m))*np.sum(A2,1)
    rho = rho.reshape(len(rho),1)
    Jsparse = np.sum(sparsityParam*np.log(sparsityParam/rho))+ \
            np.sum((1-sparsityParam)*np.log((1-sparsityParam)/(1-rho)))
    cost = Jcost + la*Jweight + beta*Jsparse

    #计算输出层的误差,稀疏矩阵的误差以及隐层误差
    D3 = -(data-A3)*dsigmoid(Z3)
    sterm = beta*(-sparsityParam/rho+(1-sparsityParam)/(1-rho))
    D2 = (np.dot(W2.T,D3)+np.repeat(sterm,m,1))*dsigmoid(Z2)

    #计算梯度
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
    #sigmoid函数
    return 1/(1+np.exp(-x))


def dsigmoid(x):
    #sigmoid导函数
    return sigmoid(x)*(1-sigmoid(x))


def computeNumericalGradient(J, theta):
    #对每一个参数进行梯度检测
    numgrad = np.zeros(theta.shape)
    epsilon = 1e-4
    n = theta.shape[0]
    E = np.eye(n)
    for i in range(0,n,1):
        delta = E[:,i]*epsilon
        numgrad[i] = (J(theta+delta)[0]-J(theta-delta)[0])/(epsilon*2.0)
    return numgrad


def checkNumericalGradient():
    #检查梯度检测函数是否正确
    x = np.array([[4],[10]])
    [value,grad] = simpleQuadraticFunction(x)
    numgrad = computeNumericalGradient(simpleQuadraticFunction,x)
    print [numgrad,grad]
    diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)
    print diff


def simpleQuadraticFunction(x):
    #设置一个简单的函数来用于检验梯度检测函数
    value = x[0]*x[0]+3*x[0]*x[1]
    grad = np.zeros((2,1))
    grad[0] = 2*x[0]+3*x[1]
    grad[1] = 3*x[0]
    return [value,grad]


if __name__ == '__main__':
    #step1 数据准备以及参数初始化
    data = data_prepare()
    theta = initiallize(25,64)
    la = 0.0001
    sparsityParam = 0.01
    beta = 3
    visible = 64
    hidden = 25

    #step2 梯度检测
    [cost,grad]=sparseAutoencoderCost(theta, visible, hidden, \
            la, sparsityParam, beta, data)
    numgrad = computeNumericalGradient(lambda (x):sparseAutoencoderCost(x, \
            visible,hidden,la,sparsityParam,beta,data),theta)
    diff = np.linalg.norm(grad-numgrad)/np.linalg.norm(grad+numgrad)
    print diff

    #step3 训练稀疏自编码
    theta = initiallize(25,64)
    [X,cost,d]=sop.fmin_l_bfgs_b(lambda (x) :sparseAutoencoderCost(x,visible, \
            hidden,la,sparsityParam,beta,data),x0=theta,maxiter=400,disp=1)
    W1 = X[0:hidden*visible].reshape(hidden,visible)
    W1 = W1.T
    #step4 W1权值图像展现
    showimg(W1,25,8)
