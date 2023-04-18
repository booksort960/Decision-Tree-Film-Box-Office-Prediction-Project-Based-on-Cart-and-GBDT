'''
Name:
        构造一棵决策树
Author:
        Booksort
Time:
        2023-3-25
Function:
        由于决策树的构造是在训练过程中搭建的，所以这也是决策树训练模块
Think:
        一份原始数据，在其他函数访问都是基于索引
        决策树的非叶子节点的都是决策变量，对于不同的取值是决策变量的取值叫做特征值

Name:
        变量：单词缩写加单词首字符大写
        函数名：单词缩写组成，第一个单词首字母小写
'''
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os


global Bs_DecTree  # 字典树
global DataSet
'''
天气：          坏-0，好-1
是否周末：       否-0，是-1
是否促销：       否-0，是-1
销量：          低-0，高-1
'''


def ImportDataset():
    DataSet = np.array((
        [0, 1, 1, 1],
        [0, 1, 1, 1],
        [0, 1, 1, 1],
        [0, 0, 1, 1],
        [0, 1, 1, 1],
        [0, 0, 1, 1],
        [0, 1, 0, 1],
        [1, 1, 1, 1],
        [1, 1, 0, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [0, 1, 1, 0],
        [1, 0, 1, 1],
        [1, 0, 1, 1],
        [1, 0, 1, 1],
        [1, 0, 1, 1],
        [1, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [1, 0, 1, 0]
    ))

    DecValCol = {"weather": 0, "day": 1, "promote": 2, "sale": 3}  # 维护决策变量的索引

    CharaValRow = list(range(DataSet.shape[0]))                                               # 集合中元数据的索引


    return DataSet, DecValCol.copy(), CharaValRow


def CountMostDivers(unionlist):  # 统计该集合中哪一类别占比最多

    return np.argmax(np.bincount(unionlist))


def calcuEntropy(colVec):  # 对于集合中的分类结果的列向量
    divNum = set(list(colVec))  # 统计分类的类别数
    numexams = len(colVec)  # 统计集合中的样本数目
    entropy = 0  # 熵值
    for a in divNum:
        elem = len([val for val in colVec if val == a])  # 统计出现指定类别的数量做分子
        prob_a = elem / numexams
        entropy = entropy - prob_a * np.log2(prob_a)

    return entropy  # 返回计算的集合的熵值


## 功能完成
def bestDecisionVarOfInformGain(dataset, decvalcol, charavalrow):  # param：多列的矩阵，包括分类结果；决策变量字典树，名字与矩阵索引
    # 统计待分类的决策变量
    decVarli = list(decvalcol.keys())
    decVarli.pop()
    datalabels = [dataset[labindex, -1] for labindex in charavalrow]    # 统计当前集合的分类结果
    # 计算当前待分类的集合的熵值
    baseEntropy = calcuEntropy(datalabels)  # 待切分的集合的熵值
    # 依次遍历计算决策变量的特征分类
    recordEntr = {}
    for a in decVarli:
        # 提取矩阵的列,[决策变量的特征值，分类结果]
        Xarray = np.array([(dataset[index, decvalcol[a]], dataset[index, -1]) for index in charavalrow])
        featVal = set(Xarray[:, 0])
        EntrSum = 0
        for feat in featVal:    # 根据决策变量不同的特征取值切分集合
            collect = np.array([Xarray[index, -1] for index in range(Xarray.shape[0]) if Xarray[index, 0] == feat])
            Entr_i = calcuEntropy(collect)
            EntrSum = EntrSum + Entr_i * (collect.size / Xarray.size)

        recordEntr[a] = EntrSum
    Gain = dict([(key, baseEntropy - val) for key, val in recordEntr.items()])
    maxGain = max(Gain.values())
    for k, v in Gain.items():
        if v >= maxGain:
            bestFeatName = k
            break
    # 统计完所有的决策变量分类的信息熵，在统计最大的信息增益
    return bestFeatName  # 返回被用于分类的决策变量的下标


def subDataSplit(dataset, featIndex, chara_val_row, var):  # 参数:带分割的数据集，依据决策变量进行分割，决策变量中的种类,列不用删除，但是行要分割
    retDataIndex = []   # 切割后要返回的数据集的索引
    for index in chara_val_row:
        if dataset[index, featIndex] == var:
            retDataIndex.append(index)

    return retDataIndex


def ConstructTree_Recursion(dataset, dec_val_col, chara_val_row):
    datalabels = [dataset[labindex, -1] for labindex in chara_val_row]           # 基于chara_val_row收集标签数据

    if len(set(datalabels)) == 1:  # set可以制作一个集合（保证集合中没有相同元素）如果集合元素数量只有1个说明data的标签都是相同的，熵值是0，可以结束递归
        #print("该集合熵值为0,分类结果为:", datalabels[0])
        return datalabels[0] == 0 and "low" or "high"
    if len(dec_val_col) == 1:  # 每建立一个节点就要删除一个特征变量，即数据集的维度变成n*1,就剩下标签维度了，那么就结束分裂
        return CountMostDivers(datalabels) == 0 and "low" or "high"  # 对于剩下的数据集，没有特征变量了，就对于数据集中的分类结果取众数作为改叶子节点的分类结果

    bestfeatName = bestDecisionVarOfInformGain(dataset, dec_val_col, chara_val_row)  # 将数据集选取最大的决策变量影响的信息增益，将数据集用决策变量的取值进行切分 返回选取的特征变量的字典名
    #print("分类的决策变量为", bestfeatName)
    decTree = {bestfeatName: {}}   # 决策字典树格式暂定

    featIndex = dec_val_col[bestfeatName]
    # 贪心算法，先算到局部最大的，就直接取走，对于子节点的兄弟节点肯恶搞并不是全局最优的
    # 删除那一列特征变量的索引，减少消耗,
    # 遍历这一层子节点的兄弟节点,即特征变量有几个取值，代表可以分成几类
    valdiv = set([dataset[index, dec_val_col[bestfeatName]] for index in chara_val_row])
    del dec_val_col[bestfeatName]

    for var in valdiv:
        subdata = subDataSplit(dataset, featIndex, chara_val_row, var)
        decTree[bestfeatName][var] = ConstructTree_Recursion(dataset, dec_val_col, subdata)

    print(decTree)

    return decTree

def DrawTree(Tree):


    return


if __name__ == "__main__":
    [dataset, Dec_Val_Col, Chara_Val_Row] = ImportDataset()
    Bs_DecTree = ConstructTree_Recursion(dataset, Dec_Val_Col, Chara_Val_Row)
    DrawTree(Bs_DecTree)
    print("result:", Bs_DecTree)

