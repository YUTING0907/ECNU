## VDSR
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
  
  越大的感受野，可使得网络能够根据更多的像素即更大的区域来预测目标像素信息，在处理大图像上有优势。文章选取3×3的卷积核，深度为D的网络拥有(2D+1)×(2D+1)的感受野，即由SRCNN的13x13变为41x41。

* 将残差residual的思想引入SR，减轻了网络的“负担”，又加速了学习速率，同时采用调整梯度裁剪(Adjustable Gradient Clipping)
  
  采用残差学习，残差图像比较稀疏，大部分值都为0或者比较小，网络不再需要学习如何恢复一张高清的HR图像，而是学习高清图像与LR图像拉伸之后的残差，因此收敛速度快。VDSR还应用了自适应梯度裁剪(Adjustable Gradient Clipping)，将梯度限制在某一范围，也能够加快收敛过程。

* 构造了一个适用于不同放大尺度的网络，并通过实验验证了该网络的可靠性
  
  随着网络结构的加深，训练成本相应增加，若网络只能做单一尺度的放大，那么网络的可重用性比较差，VDSR将不同倍数的图像混合在一起训练，这样训练出来的一个模型就可以解决不同倍数的超分辨率问题。多尺度网络的效果不仅不降低，反而提升了PSNR值。





VDSR将插值后得到的变成目标尺寸的低分辨率图像作为网络的输入，再将这个图像与网络学到的残差相加得到最终的网络的输出。


