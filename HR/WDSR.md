## WDSR
### Abstract
WDSR是基于EDSR的改进。主要有三点贡献：

1.引进残差块。 在块内增加feature map数量（即增加通道数）\
2.引入weight norm。 性能不会提高，但能使更大的学习率加快训练\
3.移除EDSR尾部沉余卷积。 提高学习的速度

### 一、Introduction

### 二、Proposal Method
#### ❤️ 2.1 第1点和第3点的贡献
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

   即：3x3 -> relu -> 3x3，卷积层的通道数依次为：\
​      3x3：input=32，output=192\
     3x3：input=192，output=32

 * WDSR-B：采用1X1的卷积核进行通道数的改变，经过relu，再使用1x1的卷积层进行通道数的改变，然后再使用3X3的卷积核进行特征提取，主要针对6-9倍的放大。

   即：1x1 -> relu -> 1x1 -> 3x3，卷积层的通道数依次为：\
​      1x1：input=32，output=192\
​      1x1：input=192，output=32\
​      3x3：input=32，output=32

  对于同样计算开销的前提下，表现性能是：WDSR-B > WDSR-A > ESDR。


* 2.另一方面是去除了很多冗余的卷积层，
  这样计算更快。如图：
  ![](https://raw.githubusercontent.com/YUTING0907/PicGo/main/img20230730171624.png)

  图中阴影部分为作者去除的冗余的两个线性卷积层（只是卷积没有激活相当于是线性变换），作者认为这些层的效果是   可以吸收到残差结构里的，通过去除实验之后，发现效果并没有下降，所以去除冗余卷积层可以降低计算开销。

#### ❤️2.2 第2点的贡献
由于Batch Normalization（BN）层在超分辨率几乎起反向作用，EDSR直接去除了BN层，而WDSR不同，另辟蹊径，提出了取代BN的Weight Normalization（WN）。

WN算是WDSR一个重要的技巧，其来自openAI在NIPS2016发表的一篇文章，就是将权重进行标准化。

作者实验得出：引入WN可以使用更高的学习速率（例如10倍）进行训练，提高训练和测试的准确性。

```
另外，BN使用的基于mini-batch的归一化统计量代替全局统计量，相当于在梯度计算中引入\
了噪声。而WN则没有这个问题，所以在生成模型，强化学习等噪声敏感的环境中WN的效果也要\
优于BN。并且，WN也没有引入额外的参数，比BN更节约显存。
```

具体的WN原理请参考这篇回答[《模型优化之Weight Normalization》](https://zhuanlan.zhihu.com/p/55102378)

参考：
[WDSR(NTIRE2018超分辨率冠军)【深度解析】](https://blog.csdn.net/leviopku/article/details/85048846)


