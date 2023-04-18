# 项目结构
## 介绍
本项目是基于kaggle上的电影数据集进行训练测试的到的电影票房预测模型

对于模型的数据处理，模型只能接受数字矩阵（int,double,float）都行，但是不能接受矩阵中有字符串

cart进行数据集划分时，是基于元素的行索引进行划分，对于特征的选择也是基于特征字典{"特征名":在矩阵的列索引}的进行查找选择的，一棵树中的数据集只有原始的一份，且不会被增删查改，希望借此减少内存的消耗

本项目中的cart是实现了树节点，是一棵链式二叉树

关于cart的预剪枝<font color=#ff9900 size=2.1 face="黑体">开发者：我也不确定cart到底有没有预剪枝技术，不同博客不同说法,我甚至无法确定到底哪种才算预剪枝，晕了</font>，我按照sklearn的输入，设置了一些参数，比如：最大层数、最大叶子节点个数，最少分裂样本数...可以通过这些参数来调节树的结构，可以有效进行预防过拟合

关于cart的后剪枝CCP，实现了具体的剪枝，但是没有进行不同alpha的交叉验证
<font color=#ff9900 size=2.1 face="黑体">开发者：懒了不愿写,可以自己写一个不同alpha的循环,调用我的后剪枝函数，可以得到小于alpha的最佳树结构，然后筛选最优树</font>

## 需要的包依赖
    numpy,matplotlib,joblib,openpyxl,pandas


## 文件功能

    ConstructDecisionTree_ID3_dict.py
一个ID3的决策树，其树结构为字典树

    train_cart.py
    train_gbdt.py

用于读取模型，开始训练

    cart_test.py
    gbdt_test.py

用于测试cart、gbdt的模型的功能是否正常


## 包功能
    --CartTree
        各个版本的cart树
    --GBDT
        只保留了一个版本的GBDT模型
    --config
        保存一些gbdt、cart的模型文件
    --dataset
        保存使用的原始数据集
    --Preteat
        是对数据集进行处理以及切割成不同的文件的功能文件
    --PlotTree
        不同版本绘制决策树的可视化文件

## 有问题或想交流的可以联系我
<font color=#ff9900 size=2.1 face="黑体">qq:2141166187</font>

<font color=#ff9900 size=2.1 face="黑体">mail:2141166187@qq.com</font>
