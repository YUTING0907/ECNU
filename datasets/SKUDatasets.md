### 1. AiProducts-Challenge(阿里2020)
OpenI平台地址: https://git.openi.org.cn/ColugoMum/Exprements-public/datasets 

数据介绍: Large-scale Product Recognition赛题与数据-天池大赛-阿里云天池

该数据集包含近 300 万张图片, 涵盖 5 万个 SKU 级商品类别. 商品图像的类别和总量均为业界之最. 此数据集中涵盖了大量的生活用品、食物等, 数据集中没有人工标注, 数据较脏, 数据分布较不均衡, 且有很多相似的商品图片.

1st-plan:[1st__Winner Solution for AliProducts Challenge Large-scale Product Recognition.pdf](https://trax-geometry.s3.amazonaws.com/cvpr_challenge/recognition_challenge_technical_reports/1st__Winner+Solution+for+AliProducts+Challenge+Large-scale+Product+Recognition.pdf)

6st-plan:[GitHub -AiProducts-Challenge](https://github.com/mingliangzhang2018/AliProducts-Challenge)

@InProceedings{Le_2020_ECCV,  
	author = {Lele Cheng and Xiangzeng Zhou and Liming Zhao and Dangwei Li and Hong Shang and Yun Zheng and Pan Pan and Yinghui Xu.},  
	title = {Weakly Supervised Learning with Side Information for Noisy Labeled Images},  
	booktitle = {The European Conference on Computer Vision (ECCV)},  
	month = {August},  
	year = {2020}  
}

### 2.RPC: 大规模零售产品结账数据集
下载地址:[Retail Product Checkout Dataset](https://www.kaggle.com/datasets/diyer22/retail-product-checkout-dataset)

OpenI平台地址:[Retail Product Checkout Dataset](https://openi.pcl.ac.cn/thomas-yanxin/RPC/datasets)

数据介绍: [RPC-PDF](https://arxiv.org/pdf/1901.07249.pdf)

@article{Wei2019RPCAL,
  title={RPC: A Large-Scale Retail Product Checkout Dataset},
  author={Xiu-Shen Wei and Quan Cui and Lei Yang and Peng Wang and Lingqiao Liu},
  journal={ArXiv},
  year={2019},
  volume={abs/1901.07249}
}

### 3. Products-10K(京东)
下载地址:[Large scale product recognition challenge](https://products-10k.github.io/challenge.html#downloads)

OpenI平台地址:[https://openi.pcl.ac.cn/thomas-yanxin/Product10k](https://openi.pcl.ac.cn/thomas-yanxin/Product10k)

数据介绍:[Products-10K](https://arxiv.org/pdf/2008.10545.pdf)

京东在线客户经常购买的10, 000种产品, 涵盖时尚、3C、食品等全品类, 医疗保健, 家居用品等. Products-10k 数据集中的所有图片均来自京东商城. 数据集中共包含 1 万个经常购买的 SKU. 所有 SKU 组织成一个层次结构. 总共有近 19 万张图片. 在实际应用场景中, 图像量的分布是不均衡的. 所有图像都由生产专家团队手工检查/标记.

1st-plan: 冠军方案分享:[ICPR 2020大规模商品图像识别挑战赛冠军解读](https://blog.51cto.com/u_15298598/3121258)

### 4. iMaterialist FGVC6 产品识别挑战赛(CVPR 2019码隆科技)
数据介绍:

该数据集共有 2, 019 个产品类别, 它们被组织成一个具有四个层次的层次结构. 此类别树可以在product_tree.json中找到, 并使用product_tree.pdf进行可视化. 每个叶节点对应一个类别 id, 类别共享同一个祖先属于同一个超类. 树结构不参与评估, 但可能在模型训练期间使用.

train.json包含id, class, url每个训练图像, 您可以在其中使用和类标签url下载相应的图像. 训练数据包含来自 2, 019 个类别的 1, 011, 532 张图像(每个类别的范围从 158 到 1050 张图像).
val.json包含id, class, url验证集中的图像. 验证数据有 10, 095 张图像(每个类别大约 5 张).
test.json包含id, url测试集中的图像. 测试数据有 90, 834 张图像(每个类别大约 45 张).

OpenI平台地址: [https://openi.pcl.ac.cn/thomas-yanxin/iMaterialist](https://openi.pcl.ac.cn/thomas-yanxin/iMaterialist)
数据下载: 比赛数据可在Google Drive或百度盘下载(密码:qecd)

1st-plan:[iMaterialist Challenge on Product Recognition](https://www.kaggle.com/c/imaterialist-product-2019/)

### 5. SmartUVM_Datasets(2019哈工大(深圳))
数据介绍:[SmartUVM_Datasets(全球新零售环境提供标准数据集).pdf](https://dl2link.com/Selected%20Journal%20Publications/Towards%20New%20Retail%20A%20Benchmark%20Dataset%20for%20Smart%20Unmanned%20Vending%20Machines.pdf)

OpenI平台地址:[https://openi.pcl.ac.cn/thomas-yanxin/SmartUVM_Datasets/datasets](https://openi.pcl.ac.cn/thomas-yanxin/SmartUVM_Datasets/datasets)

数据下载: [SmartUVM_Datasets_down(8G).tar](https://www.dl2link.com/dataset/SmartUVM_Datasets.tar)


更多参考：
https://openi.pcl.ac.cn/ColugoMum/Dataset



