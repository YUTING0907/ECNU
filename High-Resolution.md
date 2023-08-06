## super resolution高分辨率
简单来说，超分辨率方法分为三类：基于插值的方法，基于重建的方法和基于学习的方法（即深度学习方法）

传统方法中，基于插值的方法包括最近邻插值、双线性插值和双三次插值等，具有算法简单，处理速度快，但在诸如边缘、纹理等像素突变处的处理效果差，易出现锯齿和块效应；

基于重构的方法包括频域方法和空域方法，但无法很好的模拟现实场景；

本文将具体介绍深度学习图像超分辨率SR任务应先掌握的基本概念知识，然后介绍SR任务应该从哪方面入手，希望对SR能有个全局方向性的认识，能更好的把论文串起来。

### 一、SR定义
超分辨率SR的定义：将低分辨率的图像通过算法转换成高分辨率图像

SR分两种：
SISR：单图像超分辨率\
VSR：视频超分辨率

通常的超分辨率指SISR，即给定一个低分辨率（LR）图像，然后重建出一个精确的高分辨率（HR）图像。

### 二、SISR任务的概念性知识
### 2.1 SISR重建的方向
1. 力求恢复出真实可靠的细节部分，对细节要求苛刻。\
应用如：医学影像上的超分辨率重建，低分辨率摄像头人脸或者外形的恢复等。

2. 追求整体视觉效果，细节部位要求不高。\
应用如：低分辨率视频电视的恢复、相机模糊图像的恢复等。

### 2.2 SISR方法
有监督的超分方法和无监督的超分

有监督方法的基础是LR-HR图像对，网络模型的结构多种多样，下面介绍四种常见的结构。

**a.pre-upsampling SR**\
因为直接学习低分辨率图像和高分辨率图像之间的映射过程会比较困难，Dong等人在SRCNN中首次使用了pre-upsampling SR结构，即先对低分辨率图像做上采样操作，使上采样后的图像尺寸与高分辨率相同，然后学习该上采样后的图像和高分辨率图像之间的映射关系，极大地降低了学习难度。但是，预先上采样通常会带来副作用（例如，噪声放大和模糊），并且由于大多数操作是在高维空间中执行的，因此时间和空间的成本比其他框架要高得多。

![](https://raw.githubusercontent.com/YUTING0907/PicGo/main/img20230806111934.png)

**b.Post-upsampling SR**\
为了提高计算效率并充分利用深度学习技术，研究人员提出在低维空间进行大多数的运算，在网络的末端再进行上采样操作。该做法的好处是，由于具有巨大计算成本的特征提取过程仅发生在低维空间中，大大降低了计算量和空间复杂度，该框架也已成为最主流的框架之一，在近年的模型中被广泛应用。

![](https://raw.githubusercontent.com/YUTING0907/PicGo/main/img20230806111955.png)

**c.Progressive upsampling SR**\
虽然Post-upsampling SR很大程度上降低了计算难度，但对于比例因子较大的情况（4倍、8倍超分），使用Post-upsampling SR方法有较大的学习难度。而且，对于不同比例因子，需要分别训练一个单独的SR网络模型，无法满足对多尺度SR的需求。Progressive upsampling SR 框架下的模型是基于级联的CNN结构，逐步重建高分辨率图像。在每一个阶段，图像被上采样到更高的分辨率，Laplacian金字塔SR网络（LapSRN）就采用了上述框架。通过将一个困难的任务分解为简单的任务，该框架下的模型大大降低了学习难度，特别是在大比例因子的情况下，能够达到较好的学习效果。然而，这类模型也存在着模型设计复杂、训练稳定性差等问题，需要更多的建模指导和更先进的训练策略。

![](https://raw.githubusercontent.com/YUTING0907/PicGo/main/img20230806112023.png)

**d.Iterative up-and-down Sampling SR**\
为了更好地捕捉LR-HR图像对之间的相互依赖关系，在SR中引入了一种高效的迭代过程，称为反投影。DBPN就是基于该结构的模型之一，它交替连接上采样层和下采样层，并使用所有中间过程来重建最终的HR。该框架下的模型可以更好地挖掘LR-HR图像对之间的深层关系，从而提供更高质量的重建结果。然而，反投影模块的设计标准仍然不清楚，由于该机制刚刚被引入到基于深度学习的SR中，具有很大的潜力，需要进一步探索。

![](https://raw.githubusercontent.com/YUTING0907/PicGo/main/img20230806112047.png)

SR重建任务的两个优化问题：
* 如何在大尺寸放大获得更好的细节收益
* 针对Mild、Difficult现实LR图像中存在的噪声，如何在放大图像的同时不放大噪声，减弱噪声对重建的影响

### 2.3 SISR的过程步骤
基于深度学习超分辨率的起源：SRCNN(ECCV2014)

SRCNN文章于2014年提出，是第一篇将深度卷积神经网络（CNN）引入SR领域。\
作者只使用了三个卷积层，卷积核的大小分为为9x9,1x1和5x5。\
这三个卷积层也为SISR过程定义了三个步骤（后续的SISR文章都是基于这三个步骤的改进）：

* 1.LR特征提取（Patch extraction and representation），这个阶段主要是对LR进行特征提取，并将其特征表征为一些feature maps，通常称为浅层特征提取；

* 2.特征的非线性映射（Non-linear mapping），这个阶段主要是将第一阶段提取的特征映射至HR所需的feature maps；
  这部分主要用三篇论文（[VDSR](https://github.com/YUTING0907/ECNU/blob/main/HR/VDSR.md)，[EDSR](https://github.com/YUTING0907/ECNU/blob/main/HR/EDSR.md)，
[WDSR](https://github.com/YUTING0907/ECNU/blob/main/HR/WDSR.md)
)来讲解。

* 3.HR重建（Reconstruction），这个阶段是将第二阶段映射后的特征使用上采样方法恢复为HR图像。
既2016年的VDSR后，超分辨率SR的重建大部分使用ESPCN论文中提出的pixel-shuffle\
pixel-shuffle操作其实就是将H * W * C * r * r  ==>  rH * rW * C 
分辨遍历每一个通道将r与H、W混合（shuffle），即H * W 放大为 rH * rW，将照片从原来的大小放大为r倍。

```
在pytorch中：官方提供了pixel shuffle方法：
CLASS torch.nn.PixelShuffle(upscale_factor)
```


### 三、基于深度学习SISR网络的构建
上面介绍了SR的背景信息和概念知识，以下介绍SR的网络设计。

图像高分辨任务现有主流的方法还是基于监督的深度学习方法，通常数据集是SR图像或者SR-LR图像对，但是一般现实很少能收集到SR-LR对，所以主要还是SR数据集，对于只有SR数据集，会先用一定的方法生成LR数据。然后再基于网络模型生成的HR图像，与真实图像之间进行差异比较

虽然SR最流行的损失函数是逐像素均方误差(即像素损失)，但更强大的模型倾向于使用多个损失函数的组合

#### 3.1 Supervised Image Super-resolution 监督图像高分辨

* 基于卷积神经网络的方法\
1.SRCNN(2016)：第一个SR深度学习网络，Image Super-Resolution Using Deep Convolutional Networks\
2.VDSR(2016)：首次提出利用残差深度网络解决SR问题\
3.ESPCN(2016)：基于像素重排列，不需要对LR进行上采样，使用卷积的方式逐步恢复至目标分辨率大小\
3.DRCNN\
4.HAN(2016)：整体注意力网络(holistic attention network) ，考虑到了多尺度层之间的相互依赖关系，以及各层特征的通道和空间相关性，帮助网络捕获更多的信息特征\
5.EDSR/MDSR(2017)：增强的深度超分辨率(enhanced deep super-resolution，EDSR)网络\
6.RCAN(2018)：首个将注意力机制应用于SR问题的网络\
7.MSRN(2018)：多尺度残差网络 (multi-scaleresidualnetwork，MSRN)，该方法在残差块上进行改进，并加入了多尺度大小的卷积核， 实现了不同尺度图像特征的自适应地检测\
8.MSFFRN(2020)：多尺度特征融合残差模块(multi-scale feature fusion residual block，MSFFRB)，通过多个交织路 径充分利用不同尺度下的浅层和深层局部图像特征信息。
9.MSFIN(2021)：轻量级的方法，就如何让复杂的SR算法迁移至移动设备进行了研究。

* 基于GAN的方法\
当缩放因子较大时，重建的SR图像由于缩放因子较大而缺乏纹理细节，重建效果并不理想。而 GAN 具有强大的生成力，可以很好地解决该问题。\
1.SRGAN(2016)\
2.USISResNet(2020)：一种无监督的超分算法\
3.BSRGAN(2021)

* 基于自注意力transformer的方法

### 四、几类损失函数
* 1.像素损失函数 
* 2.内容损失函数：为了提升感知质量，利用神经网络中的生成的图像特征与真实图像特征之间的距离来进行计算。 
* 3.对抗损失函数 
* 4.感知损失函数：在 SRGAN 中将感知函数定义成内容损失和对抗损失的加权和

### 五、评价指标
PSNR： Peak Signal-to-Noise Ratio，峰值信噪比。PSNR数值越高，代表图像质量越好。

SSIM：结构相似性评价，Structural Similarity。SSIM是衡量两幅图像相似度的指标，其取值范围为[0,1]，SSIM的值越大，表示图像失真程度越小，说明图像质量越好。
意见平均分MOS（主观方法）：通过邀请接受过训练的普通人以及未接受过训练的普通人来对重建的图像进行评分，并且两者人数大致均衡。通过给重建图像打分，再对最后的得分进行平均，在视觉感知方面远远优于其它评价指标，可以准确测量图像感知质量。

### 六、数据集
* NTIRE Challenge
  CVPR(IEEE Conference on Computer Vision and Pattern Recognition)是世界顶级的计算机视觉会议（三大顶会之一，即IEEE国际计算机视觉与模式识别会议，另外两个是ICCV和ECCV）。
  CVPR下NTIRE（New Trends in Image Restoration and Enhancement Challenges）比赛，主要涉及图像超分辨率、图像去噪、去模糊、去摩尔纹、重建、去雾。本文主要基于NTIRE的超分辨率方面来谈，且只到2018年为止，更新的方法亲自行查阅资料。

  NTIRE主要有三个方向：图像超分辨率（super-resolution）、图像去雾（dehazing）、光谱重建（spectral reconstruction）。
  在超分辨率上有四个赛道：使用经典的bicubic（双三次插值）降尺度方式作为待重建图像，进行8倍放大重建。这也是目前大部分文献中最常见的设置方式之一。而其余三个赛道均是来自不同程度（Mild、Difficult、Wild）未知退化算子模拟相机采集的待重建图像（目的是模拟现实的图像），进行4倍放大重建。
  NTIRE2018比赛使用的数据集为DIV2K数据集，一共包含1000张2K分辨率的RGB图像，其中800张为训练集，100张为验证集，100张为测试集。评价标准使用了PSNR、SSIM。PSNR，即峰值信噪比，可以比较SR结果图和ground truth(即原高清大图)之间的差距；SSIM，即结构相似性，可以评价SR的恢复效果，更注重细节恢复。

* PIRM Challenge
  PIRM挑战是ECCV下的，其中一个子挑战关注轻量级能运用在smartphone上的研究和HR图像生成的准确率和质量
  
* DIV2K数据集下载地址：
官方：https://data.vision.ee.ethz.ch/cvl/DIV2K/

* 一般超分辨率文章使用的5个测试集Set5 , Set14, BSDS100, Urban100 and Manga109下载地址：original test datasets (HR images)
![](https://raw.githubusercontent.com/YUTING0907/PicGo/main/img20230805165034.png)

  

### 七、未来发展
未来在超分领域的改进方向，可以包括提出更复杂的损失函数；实现任意的超分辨率构建；提升性能的同时，追求轻量化；多种网络模块的有效组合；如何降低数据集图片质量，如盲超分技术来解决未知退化模型问题。

