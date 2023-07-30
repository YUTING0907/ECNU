## VDSR
### 一、Abstract
发表在CVPR2016上，
Accurate Image Super-Resolution Using Very Deep Convolutional Networks
精确的图像超分辨率使用非常深的卷积网络VDSR，CVPR2016的一篇很有创新的文章

code: https://cv.snu.ac.kr/research/VDSR/
github(caffe): https://github.com/huangzehao/caffe-vdsr
github(tensorflow): https://github.com/Jongchan/tensorflow-vdsr
github(pytorch): https://github.com/twtygqyy/pytorch-vdsr

SRCNN利用深度学习解决SISR的开山之作起到了开创性的作用，但同时存在三个问题需要进行改进：
* 依赖于小图像区域的内容；
* 训练收敛太慢；
* 网络只对于某一个比例有效。

因此VDSR在SRCNN基础上做了改进，主要有如下三点创新贡献：
* 加深了网络（20层），增加了感受野
  
  越大的感受野，可使得网络能够根据更多的像素即更大的区域来预测目标像素信息，在处理大图像上有优势。一个更深度的网络势必能带来更大的感受野，这就使得网络能够利用更多的上下文信息，能够有更全局的映射。
  文章选取3×3的卷积核，深度为D的网络拥有(2D+1)×(2D+1)的感受野，即由SRCNN的13x13变为41x41。

* 将残差residual的思想引入SR，减轻了网络的“负担”，又加速了学习速率，同时采用调整梯度裁剪(Adjustable Gradient Clipping)

  残差学习通过对LR图片学习高频细节，然后加到LR上以获得SR图像。这可以更好的理解很多作者讲的ill-posed problem，因为我们是从低频图LR上估计高频图。采用残差学习，残差图像比较稀疏，大部分值都为0或者比较小，网络不再需要学习如何恢复一张高清的HR图像，而是学习高清图像与LR图像拉伸之后的残差，因此收敛速度快。VDSR还应用了自适应梯度裁剪(Adjustable Gradient Clipping)，将梯度限制在某一范围，也能够加快收敛过程。

* 构造了一个适用于不同放大尺度的网络，并通过实验验证了该网络的可靠性
  
  随着网络结构的加深，训练成本相应增加，若网络只能做单一尺度的放大，那么网络的可重用性比较差，VDSR将不同倍数的图像混合在一起训练，这样训练出来的一个模型就可以解决不同倍数的超分辨率问题。多尺度网络的效果不仅不降低，反而提升了PSNR值。

### 二、Proposed Method
首先将图像进行插值得到ILR图像，再将其输入网络（在总结部分可看出，这是个缺陷）。网络是基于VGG19的，利用了19组conv+relu层，每个conv采用的filter规格为3*3*64。

在做卷积处理边缘像素时的三种解决办法：

1. 限制核中心与边缘的距离；

2. 用0填充，通过公式计算padding宽度；

3. 用边缘像素填充，通过公式计算padding宽度。

SRCNN用方法1导致SR缩小，VDSR采用方法2避免了此问题

### 三、Understanding Properties
本论文给我带来的三点启发性思考，本部分继续探索一下三个问题：

1. deeper networks give better performances；

2. residual-learning network converges much faster& learning rate；

3. our method with a single network performs as well as a method using multiple networks trained for each scale。

#### 3.1 The Deeper, the Better
A large receptive field means the network can use more context to predict image details.
为什么说更大的感受野意味着网络能够使用更多信息预测照片的细节信息。

**感受野**定义为：输出图像中每个像素能够反映输入图像区域的大小。
```
在卷积神经网络中，感受野的定义是卷积神经网络每一层输出的特征图（feature map）上的像素点在原始图像上映射的区域大小。  --博客园
```

针对5X5的输入图像，经过卷积核为3X3的filter，得到3X3的输出，再经过卷积核为3X3的filter，得到1X1个像素；
即此时一个像素观察到原图5X5的区域，由此可看出网络越深输出图像的单个像素的感受野越大，对图像的整体信息把握的越好。
![](https://raw.githubusercontent.com/YUTING0907/PicGo/main/img20230730141743.png)

#### 3.2 residual-learning network converges much faster& learning rate
残差学习能够收敛得非常快

#### 3.3 Single Model for Multiple Scales
![](https://raw.githubusercontent.com/YUTING0907/PicGo/main/img20230730142937.png)

用一种如scale3模型测试另一种如scale2的效果并不好，但用多种如scale2，scale3，scale4同时训练的模型（至少包括被测试图像的如scale2尺度即可）测试scale2的效果却很好，甚至超过仅用scale2训练的模型在测试scale2时的表现。

因此一个网络通过多种scale训练后的模型能用很好的表现，且能处理多尺度的放大问题。



