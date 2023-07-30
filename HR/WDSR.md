## WDSR
### Abstract
WDSR是基于EDSR的改进。主要有三点贡献：

1.引进残差块。 在块内增加feature map数量（即增加通道数）\
2.引入weight norm。 性能不会提高，但能使更大的学习率加快训练\
3.移除EDSR尾部沉余卷积。 提高学习的速度

### 一、Introduction

### 二、Proposal Method
❤️ 2.1 第1点和第3点的贡献
* 1.一方面是改造了残差块
  如图所示
  ![](https://raw.githubusercontent.com/YUTING0907/PicGo/main/img20230730165725.png)

  左图为EDSR的残差块，右图为WDSR的残差块。\
  EDSR两个卷积层的输入输出通道均相同（等矩形表示）；
  对于WDSR，作者将第一个3x3的卷积层的输入通道减小，然后输出更大的通道作为第二个卷积层的输入（relu层不改变通道数），形成通道数扩大再减少的结构（用梯形表示输入输出通道的不同）。\
  这样，可以保证相对于EDSR，在不增加参数的情况下，第一个卷积层增加了输出的通道数量，即增加feature map的宽度，获得了更多的特征信息。

  注：通道数=特征图数=滤波器数

WDSR-A和WDSR-B的区别：
 * WDSR-A：用3X3的卷积先对通道数进行增大，经过relu激活层，再使用3X3的卷积对通道数进行缩小，主要针对2-4倍的放大。

   即：3x3 -> relu -> 3x3，卷积层的通道数依次为：

​      3x3：input=32，output=192

## WDSR
### Abstract
WDSR是基于EDSR的改进。主要有三点贡献：


1.引进残差块。 在块内增加feature map数量（即增加通道数）\
2.引入weight norm。 性能不会提高，但能使更大的学习率加快训练\
3.移除EDSR尾部沉余卷积。 提高学习的速度


### 一、Introduction


### 二、Proposal Method
❤️ 2.1 第1点和第3点的贡献
* 1.一方面是改造了残差块
  如图所示
  ![](https:raw.githubusercontent.com/YUTING0907/PicGo/main/img20230730165725.png)


  The figure on the left shows the residual block of EDSR, and the figure on the right shows the residual block of WDSR. \
  The input and output channels of both convolutional layers of EDSR are the same (equal rectangle representation);
  For WDSR, the author reduces the input channel of the first 3x3 convolutional layer, and then outputs a larger channel as the input of the second convolutional layer (the relu layer does not change the number of channels), forming a structure in which the number of channels expands and then decreases (the difference between the input and output channels is represented by a trapezoid). \
  In this way, it can be guaranteed that relative to EDSR, without increasing parameters, the first convolutional layer increases the number of channels of the output, that is, increases the width of the feature map, and obtains more feature information.


  Note: Number of channels = number of feature maps = number of filters


Differences between WDSR-A and WDSR-B:
 * WDSR-A: Use 3X3 convolution to increase the number of channels first, go through the relu activation layer, and then use 3X3 convolution to reduce the number of channels, mainly for 2-4 times amplification.


   That is: 3x3 -> relu -> 3x3, the number of channels of the convolutional layer is:


​      3x3：input=32，output=192


​    3x3：input=192，output=32


 * WDSR-B: Use a 1X1 convolution kernel to change the number of channels, after relu, then use a 1x1 convolutional layer to change the number of channels, and then use a 3X3 convolution kernel for feature extraction, mainly for 6-9 times amplification.


   That is: 1x1 -> relu -> 1x1 -> 3x3, the number of channels of the convolutional layer is:
​      1x1：input=32，output=192
​      1x1：input=192，output=32
​      3x3：input=32，output=32

* 2. On the other hand, it removes a lot of redundant convolutional layers,
  This way the calculation is faster. As shown in the picture:


❤️2.2 Contribution to point 2






## WDSR
### Abstract
WDSR是基于EDSR的改进。主要有三点贡献：


1.引进残差块。 在块内增加feature map数量（即增加通道数）\
2.引入weight norm。 性能不会提高，但能使更大的学习率加快训练\
3.移除EDSR尾部沉余卷积。 提高学习的速度


### 一、Introduction


### 二、Proposal Method
❤️ 2.1 第1点和第3点的贡献
* 1.一方面是改造了残差块
  如图所示
  ![](https:raw.githubusercontent.com/YUTING0907/PicGo/main/img20230730165725.png)


  The figure on the left shows the residual block of EDSR, and the figure on the right shows the residual block of WDSR. \
  The input and output channels of both convolutional layers of EDSR are the same (equal rectangle representation);
  For WDSR, the author reduces the input channel of the first 3x3 convolutional layer, and then outputs a larger channel as the input of the second convolutional layer (the relu layer does not change the number of channels), forming a structure in which the number of channels expands and then decreases (the difference between the input and output channels is represented by a trapezoid). \
  In this way, it can be guaranteed that relative to EDSR, without increasing parameters, the first convolutional layer increases the number of channels of the output, that is, increases the width of the feature map, and obtains more feature information.


  Note: Number of channels = number of feature maps = number of filters


Differences between WDSR-A and WDSR-B:
 * WDSR-A: Use 3X3 convolution to increase the number of channels first, go through the relu activation layer, and then use 3X3 convolution to reduce the number of channels, mainly for 2-4 times amplification.


   That is: 3x3 -> relu -> 3x3, the number of channels of the convolutional layer is:


​      3x3：input=32，output=192


​    3x3：input=192，output=32


 * WDSR-B: Use a 1X1 convolution kernel to change the number of channels, after relu, then use a 1x1 convolutional layer to change the number of channels, and then use a 3X3 convolution kernel for feature extraction, mainly for 6-9 times amplification.


   That is: 1x1 -> relu -> 1x1 -> 3x3, the number of channels of the convolutional layer is:
​      1x1：input=32，output=192
​      1x1：input=192，output=32
​      3x3：input=32，output=32

* 2. On the other hand, it removes a lot of redundant convolutional layers,
  This way the calculation is faster. As shown in the picture:


❤️2.2 Contribution to point 2








 * WDSR-B：采用1X1的卷积核进行通道数的改变，经过relu，再使用1x1的卷积层进行通道数的改变，然后再使用3X3的卷积核进行特征提取，主要针对6-9倍的放大。

   即：1x1 -> relu -> 1x1 -> 3x3，卷积层的通道数依次为：
​      1x1：input=32，output=192
​      1x1：input=192，output=32
​      3x3：input=32，output=32
  
* 2.另一方面是去除了很多冗余的卷积层，
  这样计算更快。如图：

❤️2.2 第2点的贡献




