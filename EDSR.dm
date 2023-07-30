## EDSR
### Abstract
发表在CVPR2017， 
增强型深度残差网络EDSR，是NTIRE2017超分辨率挑战赛上获得冠军的方案。

github(torch): https://github.com/LimBee/NTIRE2017
github(tensorflow): https://github.com/jmiller656/EDSR-Tensorflow
github(pytorch): https://github.com/thstkdgus35/EDSR-PyTorch

1.作者推出了一种加强版本的基于Resnet块的超分方法，它实际上是在SRResnet上的改进，去除了其中没必要的的BN部分，从而在节省下来的空间下扩展模型的size来增强表现力，它就是EDSR，其取得了当时SOAT的水平。

2.此外，作者在文中还介绍了一种基于EDSR的多缩放尺度融合在一起的新结构——MDSR。

3.EDSR、MDSR在2017年分别赢得了NTIRE2017超分辨率挑战赛的冠军和亚军。

4.此外，作者通过实验证明使用L1−Loss比L2−Loss具有更好的收敛特性。
  
