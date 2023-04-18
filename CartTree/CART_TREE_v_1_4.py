'''
Name:
        构造一棵决策树的类
Author:
        Booksort
Time:
        2023-4-15
Log:

Version:
        v1.4
Function:
        搭建决策树的类
Think:
        增加cart树，
        ccp的后剪枝算法，ccp算法是需要基于独立的验证集进行剪枝（分一个10%测试集）

'''
import copy

import numpy as np


class BsNode:
    def __init__(self):
        self.leaf = False  # 表明节点的类型   False-非叶子节点，True-叶子节点
        self.parent = None  # 当前节点发父节点
        self.Left = None  # 左子树节点指针
        self.Right = None  # 右子树节点指针

        self.gini = 0  # 分类树使用，当前节点的在分类数据集中的计算当前集合的基尼系数进行保存
        self.errsquared = 0  # 回归树使用，当前节点对于数据集计算出来的所有的最小误差平法之和

        self.beforevalueIndex = []  # 在构造决策树中在 分类前 分类到该节点中的数据的索引,分类后的集合会被其他节点描述

        self.labelIndex = {}  # 还剩下的决策变量的索引
        self.decValLabel = ""  # 该节点决策变量决策的变量名,通过维护决策变量索引的字典树进行查找决策
        self.threshold = 0  # 决策分割的阈值

        self.tree_depth = 0  # 当前节点所在整棵树的层数/深度，整棵树的深度从0开始，根节点就是第0层
        self.classresult = None  # 只有叶子节点才使用的分类结果

    def get_node_in_tree_depth(self):
        return self.tree_depth


'''
需要功能：
    1、找出依次可以计算的特征变量
    2、计算决策变量对于不同选值的对于集合基尼系数
    3、对于离散值还是连续值的决策变量的取值，都要先排序，再依次计算出中值，然后计算不同中值分类的GINI系数

    sklearn中非叶子节点一定有两棵子树或者两个叶子节点

    注：需要处理调缺失值，数据集中的缺失值用-1代替，数据化后的值都是从0开始,选择均值替换缺失值

'''


class BsTree:
    def __init__(self, tree_class="regression", max_depth=0, min_sample_leaf=1, max_leaf_nodes=0, min_sample_split=0,
                 min_impurity_decrease=0):
        self._root = None
        self.leaf_sample_sum = 1  # 树中的叶子节点总数
        self._array_tree = []  # 数组树，数组中存储节点
        self.array_leaf = []  # 统计cart数的叶子节点
        self._Tree_class = ""

        self.raw_dataset = None
        self.raw_samp_row = []
        self.raw_label_dict = {}
        self.raw_labelset = np.array([])

        print("This is cart_tree v1.3")
        try:
            if tree_class != "classify" and tree_class != "Classify" and tree_class != "Regression" and tree_class != "regression":
                raise ValueError("The Class of Decision Error:tree_class" + tree_class)
            else:
                self._Tree_class = tree_class
        except ValueError as e:
            print(e)
            exit(-1)

        '''关于决策树中存在的一些限制参数
            1.形成叶子节点的最小样本数：min_sample_leaf
            2.决策树的最大层数/深度：max_depth
            3.最大叶子节点数目：max_leaf_nodes       （树中叶子节点树目等于这个就停止分割）
            4.内部节点划分的最小样本数：min_sample_split(集合样本小于这个数量就不在分割)
            5.集合样本划分最小不存度：min_impurity_decrease（集合的基尼系数小于这个值就不再进行划分）
        '''
        self._max_depth = max_depth
        self._min_sample_leaf = min_sample_leaf
        self._max_leaf_nodes = max_leaf_nodes
        self._min_sample_split = min_sample_split
        self._min_impurity_decrease = min_impurity_decrease

    def fit(self, dataSet, labelset, sampIndex, labelIndex, val_data_set, val_y):  # 通过读入数据集开始训练/构造一棵决策树,样本的索引，签值的索引
        self.raw_dataset = copy.deepcopy(dataSet)
        self.raw_labelset = copy.deepcopy(labelset)
        self.raw_samp_row = copy.deepcopy(sampIndex)
        self.raw_label_dict = copy.deepcopy(labelIndex)
        self._root = self.__ConstructDecisionTreeCycle(self.raw_dataset, self.raw_labelset, self.raw_samp_row,
                                                       self.raw_label_dict)
        self._root = self.__cart_ccp_prune(val_data_set, val_y, 0.00000003)  # 后剪枝

        print("决策树构造完成")

    def get_tree_class(self):
        return self._Tree_class

    def get_root_node(self):
        return self._root

    def get_all_node_list(self):
        return self._array_tree

    def predict(self, XdataSet):
        '''
        遍历二叉树，得到节点的labelname,阈值，小于等于阈值的去左子树，知道判断出是叶子节点
        :param XdataSet: 对于要预测的数据的输入，只有决策变量的数据没有结果,数据数量*决策变量的数量
        :return: 决策树预测的结果 一个列表
        '''
        m = XdataSet.shape[0]
        y_pred = []
        for r in range(m):
            cur = self._root
            while cur.leaf != True:
                thres = cur.threshold
                lab = cur.labelIndex[cur.decValLabel]
                if XdataSet[r, lab] <= thres:
                    cur = cur.Left
                else:
                    cur = cur.Right
            y_pred.append(cur.classresult)

        return y_pred



    def score(self, XdataSet, ylabel):
        '''
        调用predict函数预测得到结果，然后计算模型对于测试集的预测效果
        :param XdataSet: 测试集的输入数据，决策变量
        :param Ylabel: 测试集的结果数据
        :return: 正确率
        '''

        yout = self.predict(XdataSet)
        err = [yout[index] - ylabel[index] for index in range(len(yout))]
        err_score = sum(err) / (len(ylabel) * 10000000)

        return err_score, yout

    def __cart_ccp_prune(self, val_x, val_y, alpha):  # 后剪枝算法
        '''
        CCP剪枝
        传入超参数alpha（自己设置，暂时不使用交叉验证）
        对于已经训练好了的cart，再调用predict去预测验证集
        对原始树中每一个可以剪枝的非叶子节点进行剪枝，然后去测试与剪枝前误差平方和变化最小，这是位来加强模型的泛化能力
        :param val_data: 验证集的特征，来源于一半的测试集
        :param y: 验证集的输出值
        :return:
        '''

        # 使用决策树去预测验证集
        root_copy =copy.deepcopy(self._root)
        '''
            1.先要找到一棵树中所有能够被剪枝的非叶子节点，不包括根节点.
            2.然后手动去遍历，以确定哪个剪枝的节点确保被计算
            3.统计出验证集的均方差，，首先需要统计出剪枝前的均方误差，然后相剪，在处理叶子节点处减一
            4。得到所有的小于alpha的值，再统计所有变换最小的剪切节点，然后还要计算该节点的子节点，选取最大的一个进行剪切
        '''
        origin_pred_y = self.predict(val_x)
        origin_mse = np.sum([(origin_pred_y[r]-val_y[r])**2 for r in range(len(origin_pred_y))])

        no_leaf_list = [node for node in self._array_tree if node.leaf is False and node.parent is None]
        mse_list = []
        for node in no_leaf_list:
            y_pred = self._val_predict(val_x, root_copy, cut_node=node)
            mse = np.sum([(y_pred[r]-val_y[r])**2 for r in range(len(y_pred))])
            leaf_num = self._cacul_leaf_sum(node)
            T_num = self.leaf_sample_sum - leaf_num + 1
            mse_list.append(float((origin_mse - mse) / T_num))

        '''
        1.统计小于alpha的mse
        2.找到最小的mse的节点，且叶子节点数最少，并且找到小标
        3。可以靠下标找到最适合剪切的节点
        '''
        small_mse = list()
        for index in range(len(mse_list)):
            if mse_list[index] <= alpha:
                small_mse.append((mse_list[index], index))
        if len(small_mse) < 1:
            return root_copy
        print("进行后剪枝")
        min_mse_tuple = min(small_mse, key=lambda small_mse: small_mse[0])
        # 判断是否有多个最小的mse变化
        min_mse = min_mse_tuple[0]
        wait_cur_node_list = []

        for index in range(len(mse_list)):
            if mse_list[index] <= min_mse:
                wait_cur_node_list.append(index)

        leaf_node_sum = []
        for index in wait_cur_node_list:
            tmp = self._cacul_leaf_sum(no_leaf_list[index])
            leaf_node_sum.append((tmp, index))
        # 拿到符合条件的最大叶子节点的带剪切节点
        maxleaf_tuple = max(leaf_node_sum, key=lambda leaf_node_sum: leaf_node_sum[0])

        cutted_node = no_leaf_list[maxleaf_tuple[1]]

        if cutted_node.Left != None and  cutted_node.Right != None:
            cutted_node.Left = cutted_node.Right = None
        else:
            exit(-1)

        return root_copy

    def _val_predict(self, x_data, root, cut_node):

        m = x_data.shape[0]
        y_pred = []
        for r in range(m):
            cur = root
            while cur.leaf is False and cur != cut_node:
                thres = cur.threshold
                lab = cur.labelIndex[cur.decValLabel]
                if x_data[r, lab] <= thres:
                    cur = cur.Left
                else:
                    cur = cur.Right
            y_pred.append(cur.classresult)
        return y_pred

    def _cacul_leaf_sum(self, root):
        stack = list()
        stack.append(root)
        leaf_sum = 0
        while len(stack) > 0:
            cur = stack.pop()
            if cur.leaf is True:
                leaf_sum += 1
            if cur.Left != None:
                stack.append(cur.Left)
            if cur.Right != None:
                stack.append(cur.Right)

        return leaf_sum


    def __ConstructDecisionTreeCycle(self, dataSet, labelSet, sampIndex, labelIndex):
        stack = list()  # 用于处理二叉树递归的元素
        node = BsNode()
        node.parent = None
        node.beforevalueIndex = copy.deepcopy(sampIndex)
        node.labelIndex = copy.deepcopy(labelIndex)
        stack.append(node)

        while len(stack) != 0:
            cur = stack.pop()

            # curgini = self.__calcuGini(np.array([labelSet[index] for index in cur.beforevalueIndex]))  # 回归树不需要基尼系数
            print("当前节点", cur.beforevalueIndex)
            if self._Tree_class == "classify" or self._Tree_class == "Classify":
                curgini = self.__calcuGini(np.array([labelSet[index] for index in cur.beforevalueIndex]))  # 回归树不需要基尼系数
                cur.gini = round(curgini, 2)
                cur.classresult = self.__CountMostDivers(
                    [labelSet[index] for index in cur.beforevalueIndex])  # 无论能否分类都有改值
            elif self._Tree_class == "Regression" or self._Tree_class == "regression":
                curgini = self.__calcuErrSquared(labelSet, np.array(cur.beforevalueIndex))  # 回归树不需要基尼系数
                cur.errsquared = round(curgini, 2)
                cur.classresult = self.__CountAveValue(
                    [labelSet[index] for index in cur.beforevalueIndex])  # 无论能否分类都有改值
            else:
                assert False

            cur.tree_depth = cur.parent != None and cur.parent.tree_depth + 1 or 1  # 更新树的深度，树的深度加1
            # 集合元素满足分类数量的最小条件
            if curgini > self._min_impurity_decrease and len(cur.beforevalueIndex) >= self._min_sample_split \
                    and cur.tree_depth <= self._max_depth - 1 and self.leaf_sample_sum <= self._max_leaf_nodes - 1 \
                    and len(cur.labelIndex) > 1 and cur.labelIndex.__len__() > 1:
                # 样本集合大于最小不纯度，数量大于最小分割数量，树的深度满足至少可以生成一个节点 满足这些条件再去分割
                best_tuple = self.__bestDecisionValOfInformGain(dataSet, labelSet, cur.beforevalueIndex,
                                                                cur.labelIndex)  # 返回：基尼系数，阈值，决策变量名
                # 新建节点
                # if self._Tree_class == "classify" or self._Tree_class == "Classify":
                #     cur.gini = best_tuple[0]
                # elif self._Tree_class == "Regression" or self._Tree_class == "regression":
                #     cur.errsquared = best_tuple[0]
                # else:
                #     assert False
                # best_tuple[0]是最大gini增益的值
                cur.threshold = best_tuple[1]
                cur.decValLabel = best_tuple[2]
                # cur.tree_depth = cur.parent != None and cur.parent.tree_depth + 1 or 1  # 更新树的深度，树的深度加1
                left, right = self.__subDataSplit(dataSet, cur.beforevalueIndex, labelIndex, cur.decValLabel,
                                                  cur.threshold)
                if self._min_sample_leaf > 0 and len(left) >= self._min_sample_leaf and len(
                        right) >= self._min_sample_leaf:
                    # 满足叶子节点的最小结合数量，开始分裂
                    self.leaf_sample_sum += 1
                    lnode = BsNode()
                    rnode = BsNode()
                    # 左叶子
                    lnode.beforevalueIndex = copy.deepcopy(left)
                    lnode.parent = cur
                    cur.Left = lnode
                    lnode.labelIndex = copy.deepcopy(lnode.parent.labelIndex)
                    if len(set([dataSet[index, labelIndex[lnode.parent.decValLabel]] for index in
                                lnode.beforevalueIndex])) <= 1:  # 如果这父节点的分裂特征变量中在该节点设于的元素的输出值唯一，那么后面就不用考虑这个特征
                        del lnode.labelIndex[lnode.parent.decValLabel]
                    # 右叶子
                    rnode.beforevalueIndex = copy.deepcopy(right)
                    rnode.parent = cur
                    cur.Right = rnode
                    rnode.labelIndex = copy.deepcopy(rnode.parent.labelIndex)
                    if len(set([dataSet[index, labelIndex[rnode.parent.decValLabel]] for index in
                                rnode.beforevalueIndex])) <= 1:
                        del rnode.labelIndex[rnode.parent.decValLabel]
                    # 入栈
                    stack.append(rnode)
                    stack.append(lnode)

                else:
                    # 不能分类，cur节点为叶子节点
                    cur.leaf = True
            else:
                # 本省就要作为叶子节点
                cur.leaf = True

            self._array_tree.append(cur)  # 放入一个列表中方便查找
            if cur.leaf == True:
                self.array_leaf.append(cur)  # 统计数的叶子节点

        return node

    def __bestDecisionValOfInformGain(self, dataSet, labelSet, sampIndex, labelIndex):
        '''
            以此遍历所提供的所有的决策变量的索引
            然后调用__calcuGini进行计算，然后找出最大的基尼系数
        :return:
        '''
        labelList = list(labelIndex.keys())
        labelList.pop()
        tuple_list = []
        for label in labelList:
            # 提取每个决策变量的列,计算获得每个决策变量的基尼系数与阈值
            thres_tuple = self.__selctThreshold(dataSet, labelSet, sampIndex, labelIndex, label)
            thres_tuple = thres_tuple + (label,)
            tuple_list.append(thres_tuple)
        min_tuple = min(tuple_list, key=lambda tuple_list: tuple_list[0])  # 找到最小的基尼系数是由哪个决策变量分割的

        return min_tuple

    def __selctThreshold(self, dataSet, labelSet, vecindex, labelIndex, col):
        '''
        :return: 返回特征值的阈值和最小的GINII系数 或 均方差
        '''
        vector_tuple = [(dataSet[index, labelIndex[col]], labelSet[index]) for index in vecindex]

        '''缺失值处理'''
        if -1 in [val[0] for val in vector_tuple]:
            num = 0
            sum = 0
            for val in [valu[0] for valu in vector_tuple]:
                if val != -1:
                    num += val
                    sum += 1
            for index in range(len(vector_tuple)):
                if vector_tuple[index][0] == -1:
                    vector_tuple[index] = (num / sum, vector_tuple[index][1])

        vectors = np.array(sorted(vector_tuple, key=lambda vector_tuple: vector_tuple[0]))
        midvalue = [(vectors[index, 0] + vectors[index + 1, 0]) / 2 for index in range(vectors.shape[0] - 1)]
        midthres = set(midvalue)
        recordScore = []
        for thres in midthres:

            if self._Tree_class == "classify" or self._Tree_class == "Classify":
                xarray1 = np.array([labelSet[index] for index in vecindex
                                    if thres >= dataSet[index, labelIndex[col]]])
                xarray2 = np.array([labelSet[index] for index in vecindex
                                    if dataSet[index, labelIndex[col]] > thres])
                gini1 = self.__calcuGini(xarray1)
                gini2 = self.__calcuGini(xarray2)
                l1 = xarray1.size / (xarray1.size + xarray2.size)
                l2 = xarray2.size / (xarray1.size + xarray2.size)
                sum_score = l1 * gini1 + l2 * gini2  # 需求最小和要求基尼增益最大一样的效果

            elif self._Tree_class == "Regression" or self._Tree_class == "regression":
                xarray1 = np.array([index for index in vecindex if thres >= dataSet[index, labelIndex[col]]])
                xarray2 = np.array([index for index in vecindex if dataSet[index, labelIndex[col]] > thres])

                squared1 = xarray1.size != 0 and self.__calcuErrSquared(labelSet, xarray1) or 0
                squared2 = xarray2.size != 0 and self.__calcuErrSquared(labelSet, xarray2) or 0
                sum_score = (squared1 + squared2)  # MSE需求最小

            else:
                assert False

            recordScore.append((sum_score, thres))

        min_tuple = min(recordScore, key=lambda recordScore: recordScore[0])

        return min_tuple

    def __calcuGini(self, colvec):
        '''
            在决策变量的取值中找出对集合的最大分类的决策中值，先对决策变量的取值进行排序，然后再依次计算中值
            相继以此中值分割作为预阈值条件，然后计算来个切分后集合基尼系数
        :return:阈值和基尼系数
        '''
        sumgini = 1
        divNum = set(list(colvec))  # 统计分类的类别数
        numexams = colvec.size  # 统计集合中的样本数目
        for a in divNum:
            elem = len([val for val in colvec if val == a])  # 统计出现指定类别的数量做分子
            prob_a = elem / numexams
            sumgini -= prob_a ** 2

        return sumgini

    def __calcuErrSquared(self, labelSet, rowvec):
        # 先计算得到这个向量对应的值的均值，在依次计算差的平方和
        ave_val = sum([labelSet[index] for index in rowvec]) / rowvec.size
        sum_square = sum([(labelSet[index] - ave_val) ** 2 for index in rowvec])

        return sum_square

    def __subDataSplit(self, dataSet, sampIndex, labelIndex, Label, thres):  # 按照计算第阈值分割集合
        left = []
        right = []
        for index in sampIndex:
            if dataSet[index, labelIndex[Label]] <= thres:
                left.append(index)
            else:
                right.append(index)

        return left, right

    def __CountMostDivers(self, unionlist):  # 返回集合中最多的元素
        return np.argmax(np.bincount(unionlist))

    def __CountAveValue(self, regresslist):  # 计算每个叶子节点作为回归时的输出的均值
        return sum(regresslist) / regresslist.__len__()

    def CycleArrayNode(self):
        for node in self._array_tree:
            print(node.tree_depth, node.leaf, node.classresult)
