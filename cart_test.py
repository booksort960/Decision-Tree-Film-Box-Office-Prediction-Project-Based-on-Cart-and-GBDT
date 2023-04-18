

import numpy as np
# from DecisionTreeClass import BsTree
from DecisionTree.CartTree.CART_TREE_v_1_4 import BsTree
from DecisionTree.PlotTree.PlotTree_v_1_2 import PltTree






def testcart():
    DataSet = np.array((
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [1, 0, 0, 0, 1],
        [2, 1, 0, 0, 1],
        [2, 2, 1, 0, 1],
        [2, 1, 1, 1, 0],
        [1, 2, 1, 1, 1],
        [0, 1, 2, 0, 0]
    ))
    CharaValRow = list(range(DataSet.shape[0]))
    DecValCol = {"outlook": 0, "temperature": 1, "humidity": 2, "windy": 3, "play": 4}  # 维护决策变量的索引
    mytree = BsTree(tree_class="regression", max_depth=4, min_sample_leaf=1, max_leaf_nodes=4, min_sample_split=0,
                    min_impurity_decrease=0)
    # mytree.fit(dataSet=DataSet[:, :-1], labelset=DataSet[:, -1], sampIndex=CharaValRow, labelIndex=DecValCol, val_data_set=val_data_set, val_y=val_y)
    mytree.CycleArrayNode()
    return

def testplot():

    DataSet = np.array((
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [1, 0, 0, 0, 1],
        [2, 1, 0, 0, 1],
        [2, 2, 1, 0, 1],
        [2, 1, 1, 1, 0],
        [1, 2, 1, 1, 1],
        [0, 1, 2, 0, 0]
    ))
    val_data_set = np.array(
        ([1, 0, 0, 0, 1],
        [2, 1, 0, 0, 1])
    )
    CharaValRow = list(range(DataSet.shape[0]))
    DecValCol = {"outlook": 0, "temperature": 1, "humidity": 2, "windy": 3, "play": 4}  # 维护决策变量的索引
    mytree = BsTree(tree_class="classify", max_depth=4, min_sample_leaf=1, max_leaf_nodes=4, min_sample_split=0,
                    min_impurity_decrease=0)
    mytree.fit(dataSet=DataSet[:, :-1], labelset=DataSet[:, -1], sampIndex=CharaValRow, labelIndex=DecValCol,
               val_data_set=val_data_set[:, :-1], val_y=val_data_set[:, -1])


    myplottree = PltTree(mytree)
    myplottree.draw()

    return

if __name__ == "__main__":
    #testdata()
    #testcart()


    testplot()
    print()



