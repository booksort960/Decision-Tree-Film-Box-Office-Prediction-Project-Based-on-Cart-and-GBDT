'''
Name:
        构造一棵决策树的类
Author:
        Booksort
Time:
        2023-4-5
Log:
        20230407-20:00~20230408-00:50: cart决策树训练模块完成包括5个预剪枝手段
        20230408-12:00~20230408-20:50: cart回归树构建完成，基于MSE进行特征分割
Version:
        v.1.2
Function:
        搭建决策树的类
Think:
        建立一个node节点类，以及树类
        基于cart算法构建的决策树应该是一个二叉树，对于节点处理连续值与离散值，而离散值也是要按照连续值处理，
        对于节点保存决策变量的分类标签也就死名字，代表该节点会使用哪个决策变量进行分类，还要保存分割的阈值，小于等阈值的进入左子树，大于阈值的进入右子树
        对于节点类中，仅需完成和节点操作相关功能API，对于cart树类，才是需要完成分类，选择计算等等功能API

        对于数据集的规定：
            N*M维矩阵，全部数字化，
            行，N,表示一个元素数据
            列，M,表示决策变量名及分类标签名，以及各自在矩阵中索引

        在决策树构造/训练过程中：
            对于当前节点待分类要从上一节点中获得数据集和以及剩余的决策变量
            对于构建的树类，功能有递归构建决策树，计算当前集合的基尼系数，在递归过程中选择基尼系数最小的决策变量构建节点
            sklearn有两种存储决策树的方案：1，数组-这样就构造了一个堆；2，数组树（选择）
                   有两种构造决策树的方案：1，使用堆（优先级队列）；2，对大栈，用于选择最优解点进行构造

            决定使用 数组树 结构来做决策树的物理存储模型，使用栈配合循环非递归来构造二叉树，怕爆栈的问题
'''
import copy

import numpy as np



class BsNode:
    def __init__(self):

        self.leaf = False      # 表明节点的类型   False-非叶子节点，True-叶子节点
        self.parent = None     # 当前节点发父节点
        self.Left = None       # 左子树节点指针
        self.Right = None      # 右子树节点指针

        self.gini = 0          # 分类树使用，当前节点的在分类数据集中的计算当前集合的基尼系数进行保存
        self.errsquared = 0    # 回归树使用，当前节点对于数据集计算出来的所有的最小误差平法之和

        self.beforevalueIndex = []      # 在构造决策树中在 分类前 分类到该节点中的数据的索引,分类后的集合会被其他节点描述

        self.labelIndex = {}        # 还剩下的决策变量的索引
        self.decValLabel = ""       # 该节点决策变量决策的变量名,通过维护决策变量索引的字典树进行查找决策
        self.threshold = 0          # 决策分割的阈值

        self.tree_depth = 0         # 当前节点所在整棵树的层数/深度，整棵树的深度从0开始，根节点就是第0层
        self.classresult = None     # 只有叶子节点才使用的分类结果


'''
需要功能：
    1、找出依次可以计算的特征变量
    2、计算决策变量对于不同选值的对于集合基尼系数
    3、对于离散值还是连续值的决策变量的取值，都要先排序，再依次计算出中值，然后计算不同中值分类的GINI系数
    
    sklearn中非叶子节点一定有两棵子树或者两个叶子节点
    
    注：需要处理调缺失值，数据集中的缺失值用-1代替，数据化后的值都是从0开始,选择均值替换缺失值
    
'''

class BsTree:
    def __init__(self, tree_class="regression", max_depth=0, min_sample_leaf=1, max_leaf_nodes=0, min_sample_split=0, min_impurity_decrease=0):
        self._root = None
        self._leaf_sample_sum = 1   # 树中的叶子节点总数
        self._array_tree = []       # 数组树，数组中存储节点
        self.array_leaf = []       # 统计cart数的叶子节点
        self._Tree_class = ""
        self.dataSet = None
        print("This is cart_tree v1.2")
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

    def fit(self, dataSet, sampIndex, labelIndex):   # 通过读入数据集开始训练/构造一棵决策树,样本的索引，签值的索引
        self.dataSet = copy.deepcopy(dataSet)
        self._root = self.__ConstructDecisionTreeCycle(self.dataSet, sampIndex, labelIndex)
        self.__cartprune()     # 后剪枝

        print("决策树构造完成")


    def predict(self, XdataSet):
        '''
        遍历二叉树，得到节点的labelname,阈值，小于等于阈值的去左子树，知道判断出是叶子节点
        :param XdataSet: 对于要预测的数据的输入，只有决策变量的数据没有结果,数据数量*决策变量的数量
        :return: 决策树预测的结果 一个列表
        '''
        cur = self._root
        while cur.leaf != True:
            thres = cur.threshold
            lab = cur.labelIndex[cur.decValLabel]
            if XdataSet[lab] <= thres:
                cur = cur.Left
            else:
                cur = cur.Right

        return cur.classresult


    def score(self, XdataSet, Ylabel):
        '''
        调用predict函数预测得到结果，然后计算模型对于测试集的预测效果
        :param XdataSet: 测试集的输入数据，决策变量
        :param Ylabel: 测试集的结果数据
        :return: 正确率
        '''
        errlist = []
        for index in range(len(Ylabel)):
            yout = self.predict(XdataSet[index, :])
            errlist.append(yout)




        return errlist

    def __cartprune(self):     # 后剪枝算法
        '''
        CCP剪枝
        Think:
                可以获得保存所有叶子节点的列表，叶子节点的父节点是没有必要删除器子树的
        :return:
        '''

        return

    def __ConstructDecisionTreeCycle(self, dataSet, sampIndex, labelIndex):
        stack = list()  # 用于处理二叉树递归的元素
        node = BsNode()
        node.parent = None
        node.beforevalueIndex = sampIndex
        node.labelIndex = labelIndex
        stack.append(node)

        while len(stack) != 0:
            cur = stack.pop()

            curgini = self.__calcuGini(np.array([dataSet[index, -1] for index in cur.beforevalueIndex]))
            print("当前节点", [dataSet[index, -1] for index in cur.beforevalueIndex])
            if self._Tree_class == "classify" or self._Tree_class == "Classify":
                cur.classresult = self.__CountMostDivers([dataSet[index, -1] for index in cur.beforevalueIndex])  # 无论能否分类都有改值
            elif self._Tree_class == "Regression" or self._Tree_class == "regression":
                cur.classresult = self.__CountAveValue([dataSet[index, -1] for index in cur.beforevalueIndex])    # 无论能否分类都有改值
            else:
                assert False

            cur.tree_depth = cur.parent != None and cur.parent.tree_depth + 1 or 1  # 更新树的深度，树的深度加1
            # 集合元素满足分类数量的最小条件
            if curgini > self._min_impurity_decrease and len(cur.beforevalueIndex) >= self._min_sample_split \
                    and cur.tree_depth <= self._max_depth - 1 and self._leaf_sample_sum <= self._max_leaf_nodes - 1 \
                    and len(cur.labelIndex) > 1 and cur.labelIndex.__len__() > 1:
                # 样本集合大于最小不纯度，数量大于最小分割数量，树的深度满足至少可以生成一个节点 满足这些条件再去分割
                best_tuple = self.__bestDecisionValOfInformGain(dataSet, cur.beforevalueIndex, cur.labelIndex)  # 返回：基尼系数，阈值，决策变量名
                # 新建节点
                if self._Tree_class == "classify" or self._Tree_class == "Classify":
                    cur.gini = best_tuple[0]
                elif self._Tree_class == "Regression" or self._Tree_class == "regression":
                    cur.errsquared = best_tuple[0]
                else:
                    assert False
                cur.threshold = best_tuple[1]
                cur.decValLabel = best_tuple[2]
                #cur.tree_depth = cur.parent != None and cur.parent.tree_depth + 1 or 1  # 更新树的深度，树的深度加1
                left, right = self.__subDataSplit(dataSet, cur.beforevalueIndex, labelIndex, cur.decValLabel,
                                                  cur.threshold)
                if self._min_sample_leaf > 0 and len(left) >= self._min_sample_leaf and len(right) >= self._min_sample_leaf:
                    # 满足叶子节点的最小结合数量，开始分裂
                    self._leaf_sample_sum += 1
                    lnode = BsNode()
                    rnode = BsNode()
                    # 左叶子
                    lnode.beforevalueIndex = left
                    lnode.parent = cur
                    cur.Left = lnode
                    lnode.labelIndex = copy.deepcopy(lnode.parent.labelIndex)
                    if len(set([dataSet[index, labelIndex[lnode.parent.decValLabel]] for index in
                            lnode.beforevalueIndex])) <= 1:
                        del lnode.labelIndex[lnode.parent.decValLabel]
                    # 右叶子
                    rnode.beforevalueIndex = right
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

            self._array_tree.append(cur)    # 放入一个列表中方便查找
            if cur.leaf == True:
                self.array_leaf.append(cur)    #统计数的叶子节点

        return node




    def __bestDecisionValOfInformGain(self, dataSet, sampIndex, labelIndex):
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
            thres_tuple = self.__selctThreshold(dataSet, sampIndex, labelIndex, label)
            thres_tuple = thres_tuple + (label, )
            tuple_list.append(thres_tuple)
        min_tuple = min(tuple_list, key=lambda tuple_list: tuple_list[0])   # 找到最小的基尼系数是由哪个决策变量分割的

        return min_tuple

    def __selctThreshold(self, dataSet, vecindex, labelIndex, col):
        '''
        :return: 返回许纳泽的阈值和最小的GINI系数
        '''
        vector_tuple = [(dataSet[index, labelIndex[col]], dataSet[index, -1]) for index in vecindex]

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
        midvalue = [(vectors[index, 0]+vectors[index+1, 0])/2 for index in range(vectors.shape[0]-1)]
        midthres = set(midvalue)
        recordScore = []
        for thres in midthres:

            if self._Tree_class == "classify" or self._Tree_class == "Classify":
                xarray1 = np.array([dataSet[index, -1] for index in vecindex
                                    if thres >= dataSet[index, labelIndex[col]]])
                xarray2 = np.array([dataSet[index, -1] for index in vecindex
                                    if dataSet[index, labelIndex[col]] > thres])
                gini1 = self.__calcuGini(xarray1)
                gini2 = self.__calcuGini(xarray2)
                l1 = xarray1.size / (xarray1.size + xarray2.size)
                l2 = xarray2.size / (xarray1.size + xarray2.size)
                sum_score = l1 * gini1 + l2 * gini2
            elif self._Tree_class == "Regression" or self._Tree_class == "regression":
                xarray1 = np.array([index for index in vecindex if thres >= dataSet[index, labelIndex[col]]])
                xarray2 = np.array([index for index in vecindex if dataSet[index, labelIndex[col]] > thres])

                squared1 = xarray1.size != 0 and self.__calcuErrSquared(dataSet, xarray1, labelIndex[col]) or 0
                squared2 = xarray2.size != 0 and self.__calcuErrSquared(dataSet, xarray2, labelIndex[col]) or 0
                sum_score = (squared1 + squared2) / (xarray1.size + xarray2.size)

            else:
                assert False

            recordScore.append((sum_score, thres))


        min_tuple = min(recordScore,key=lambda recordScore: recordScore[0])

        return min_tuple

    def __calcuGini(self, colvec):
        '''
            在决策变量的取值中找出对集合的最大分类的决策中值，先对决策变量的取值进行排序，然后再依次计算中值
            相继以此中值分割作为预阈值条件，然后计算来个切分后集合基尼系数
        :return:阈值和基尼系数
        '''
        sumgini = 1
        divNum = set(colvec)  # 统计分类的类别数
        numexams = colvec.size  # 统计集合中的样本数目
        for a in divNum:
            elem = len([val for val in colvec if val == a])  # 统计出现指定类别的数量做分子
            prob_a = elem / numexams
            sumgini -= prob_a**2

        return sumgini

    def __calcuErrSquared(self, dataSet, rowvec, colindex):
        # 先计算得到这个向量对应的值的均值，在依次计算差的平方和
        ave_val = sum([dataSet[index, -1] for index in rowvec]) / rowvec.size
        sum_square = sum([(dataSet[index, -1] - ave_val)**2 for index in rowvec])

        return sum_square


    def __subDataSplit(self, dataSet, sampIndex, labelIndex, Label, thres): # 按照计算第阈值分割集合
        left = []
        right = []
        for index in sampIndex:
            if dataSet[index, labelIndex[Label]] <= thres:
                left.append(index)
            else:
                right.append(index)

        return left, right

    def __CountMostDivers(self, unionlist):     # 返回集合中最多的元素
        return np.argmax(np.bincount(unionlist))

    def __CountAveValue(self, regresslist):     # 计算每个叶子节点作为回归时的输出的均值
        return sum(regresslist) / regresslist.__len__()

    def CycleArrayNode(self):
        for node in self._array_tree:
            print(node.tree_depth, node.leaf, node.classresult)