#UFLDL Excercise by python
-----

什么!不知道UFLDL是什么?移步http://deeplearning.stanford.edu/wiki/index.php/UFLDL_Tutorial  

Update:2016-02-09  python2.7测试通过  

##实验准备

numpy:Python的一种开源的数值计算扩展,用于进行矩阵运算.  
scipy:Python科学计算包.  
OpenBLAS:建议用先安装OpenBLAS或者MKL,然后再编译安装numpy和scipy,不然矩阵运算会很慢,https://github.com/xianyi/OpenBLAS.  

##目录结构

```
├── convolvePool
│   ├── cnn.py
│   ├── resultFeatures.mat
│   ├── stlTestSubset.mat
│   └── stlTrainSubset.mat
├── LICENSE
├── linearAutoencoder
│   ├── linearAutoencoder.py
│   └── stlSampledPatches.mat
├── MNIST
│   ├── t10k-images-idx3-ubyte
│   ├── t10k-labels-idx1-ubyte
│   ├── train-images-idx3-ubyte
│   └── train-labels-idx1-ubyte
├── README.md
├── selfLearn
│   └── selfLearn.py
├── softmax
│   └── softmax.py
├── sparseAutoencoder
│   ├── IMAGES.mat
│   └── sparseAutoencoder.py
└── stackedAutoencoder
    └── stackedAutoencoder.py
```

##章节对应
sparseAutoencoder : 稀疏自编码器  
softmax : Softmax回归  
selfLearn : 自我学习与无监督特征学习  
stackedAutoencoder : 建立分类用深度网络  
linearAutoencoder : 自编码线性解码器  
convolvePool : 处理大型图像  

##一些建议

1.千万不要跳过梯度检测的步骤,本人开始实验的时候吃了不少亏.  
2.实验中的最优化方法都采用了lbfgs,有兴趣的可以深入了解.  
3.linearAutoencoder练习中的meanPatch是对每一个输入维进行平均,但我在开始实验时对每一个图像进行平均也得到了正确的权值图像.  
4.stackedAutoencoder练习中第二层神经网络的权值图像化需要与第一层的权值作矩阵乘法.  


##附

本实验参考了matlab版的练习答案,详见http://www.cnblogs.com/tornadomeet/tag/Deep%20Learning/  
这篇博客的讲解更为详细,苦于网上没有完整的python版本的练习,因此产生了这个python版的练习答案,收获还是颇多.  

即便用了OpenBLAS编译numpy,python的矩阵运算效率依旧低于matlab,不知道用MKL编译会不会好些.


有问题可以联系本人邮箱qiqipipioioi@qq.com
