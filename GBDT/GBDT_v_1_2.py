'''
Name:
        构造一棵决策树的类
Author:
        Booksort
Time:
        2023-4-9
Log:
        20230409-15:30 ~ 20230409-21:52 : GBDT基本训练、预测模块已经基本测试完成，还需要真实的数据集进行验证模型是否正确
        20230409-21:52 : 还需要cart的CCP后剪枝算法待完成，以抑制过拟合
Function:
        利用cart决策树搭建GBDT
        可以考虑在负梯度后加一个w_i当前节点值 以抑制过拟合，猜的
'''
import numpy as np
import random
import copy
from DecisionTree.CartTree.CART_TREE_v_1_4 import BsTree
from  PlotTree.PlotTree_v_1_3 import PltTree

class BsGBDT:


    def __init__(self, learning_rate=0.1, n_estimators=100, subsample=0.8,  max_depth=4, min_sample_leaf=1, max_leaf_nodes=4, min_sample_split=0, min_impurity_decrease=0):
        '''

        :param learning_rate: 前向分布算法的学习率
        :param n_estimators: 基学习器的个数（cart树的个数）
        :param subsample: 构建cart树时的特征变量的选择率[0,1]
        '''
        self._learning_rate = learning_rate
        self._n_estimators = n_estimators
        self._subsample = subsample

        '''对于样本集GBDT中维护一个样本数据矩阵即可，通过行索引与列索引来访问,在训练时传递参数'''
        self._dataSet = np.array([])
        self._label_set = np.array([])
        self._elemRowIndex = []
        self._featColDict = {}

        '''通过一个list来维护GBDT中建立的cart树,list中的数据类型<BsTree>
            当预测时，遍历这个list，然后依次调用BsTree.traincyc即可
        '''
        self._cart_tree_list = []

        '''构建cart决策树需要的参数'''
        self._tree_class = "regression"
        self._max_depth = max_depth
        self._min_sample_leaf = min_sample_leaf
        self._max_leaf_nodes = max_leaf_nodes
        self._min_sample_split = min_sample_split
        self._min_impurity_decrease = min_impurity_decrease



    def fit(self, dataSet, Y_train, elemRowIndex, featColDict, X_val, Y_val, alpha_thres):
        '''
        训练时主要逻辑顺序
        1.根据_subsample随机挑选特征进行构建cart
        2.根据数据集的样本和预测的值取计算损失函数L，同时计算最小负梯度得到新医改决策树的驯良样本，同时也是计算残差
        3.构建新的决策树，使用上一个决策树提供的数据集进行训练
        4.直到满足终止条件，损失值小于一定阈值/决策树数量足够
        :return:
        '''
        # 1.数据集参数导入
        self._dataSet = copy.deepcopy(dataSet)
        self._label_set = copy.deepcopy(Y_train)
        self._elemRowIndex = copy.deepcopy(elemRowIndex)
        self._featColDict = copy.deepcopy(featColDict)

        all_err_square = []
        # 2.开始构建训练cart回归树
        new_labelset = self._label_set
        new_featdict = self._featColDict
        for treeNum in range(self._n_estimators):
            # 3.根据_subsample抽取一定的特征/决策变量
            # if self._featColDict.__len__() <= 1:
            #     print("特征抽取完毕,退出")
            #     break
            new_featdict = self.__subRateChara()
            err_square = []
            # 4.开始构造cart树
            print("cart树:", treeNum)
            curTree = BsTree(self._tree_class, self._max_depth, self._min_sample_leaf, self._max_leaf_nodes, self._min_sample_split, self._min_impurity_decrease)
            self._cart_tree_list.append(curTree)    # 记录
            curTree.fit(self._dataSet, new_labelset, self._elemRowIndex, new_featdict, X_val, Y_val)

            # 5.统计cart数的叶子节点的输出值，然后计算损失值/残差
            tmp_dataset = copy.deepcopy(self._label_set)
            sum_err = 0
            for leaf in curTree.array_leaf:
                # 得到叶子节点的预测值，即这个叶子节点的所有的训练数据计算残差,损失值是误差平方
                leaf_pred_out = leaf.classresult # sum([curTree.raw_labelset[index] for index in leaf.beforevalueIndex]) / len(leaf.beforevalueIndex)
                # 统计各个节点的残差，作为下一颗树的训练集

                for index in leaf.beforevalueIndex:
                    sum_err += (leaf_pred_out - curTree.raw_labelset[index])**2
                    tmp_dataset[index] = self._learning_rate * (curTree.raw_labelset[index] - leaf_pred_out)    # 浮梯度*学习率

                mse_err = sum_err / len(leaf.beforevalueIndex)  # 这是一个叶子节点mse
            # 给下一棵进行修改训练集
                err_square.append(mse_err)  # 这是整棵树的

            all_err_square.append(err_square)   # 这是所有数的

            new_labelset = tmp_dataset
            print("模型误差", sum(err_square))

            if sum(err_square) <= alpha_thres:
                break

        print("GBDT构造完成")
        myplottree = PltTree(self._cart_tree_list[-1])
        myplottree.draw()
        return 0

    def predict(self, xdata):
        # 遍历cart树列表，输入值，然后入xdata，一次将得到的值相加就是预测值
        predlist = []
        m, n = xdata.shape
        retlist = []

        for tree in self._cart_tree_list:
            predlist.append(tree.predict(xdata))
        pred = np.array(predlist)
        ret_pred = []
        for c in range(pred.shape[1]):
            ret_pred.append(sum([pred[r, c] for r in range(pred.shape[0])]))


        return ret_pred


    def score(self, xdata, ylabel):
        # 对测试集进行打分
        errlist = []
        # for a in range(xdata.shape[0]):
        yout = self.predict(xdata)
        err = [yout[index] - ylabel[index] for index in range(len(yout))]
        err_score = sum(err)/(len(ylabel) * 10000000)

        return err_score, yout


    def __subRateChara(self):
        # 放回采样
        numsamp = int((self._featColDict.items().__len__() - 1)*self._subsample + 0.5)
        if numsamp < 1:
           numsamp = 1

        print("抽取了特折数量：", numsamp)
        tmplist = list(self._featColDict.keys())
        label = tmplist.pop()
        featlist = random.sample(tmplist, numsamp)
        retdict = {}
        for a in featlist:
            retdict[a] = self._featColDict[a]
            # del self._featColDict[a]
        retdict[label] = self._featColDict[label]
        return retdict