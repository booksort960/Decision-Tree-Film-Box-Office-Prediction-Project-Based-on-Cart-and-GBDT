from DecisionTree.GBDT.GBDT_v_1_2 import BsGBDT
import numpy as np


if __name__ == "__main__":
    DataSet = np.array((
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [1, 0, 0, 0, 1],
        [2, 1, 0, 0, 1],
        [2, 2, 1, 0, 1],
        [2, -1, 1, 1, 0],
        [1, 2, 1, 1, 1],
        [0, 1, 2, 0, 0]
    ))
    CharaValRow = list(range(DataSet.shape[0]))
    DecValCol = {"outlook": 0, "temperature": 1, "humidity": 2, "windy": 3, "play": 4}  # 维护决策变量的索引

    mygbdt = BsGBDT(learning_rate=0.1,  n_estimators=101, subsample=1, max_depth=4, min_sample_leaf=1, max_leaf_nodes=4, min_sample_split=1, min_impurity_decrease=0)
    mygbdt.fit(dataSet=DataSet, elemRowIndex=CharaValRow, featColDict=DecValCol)
    print(mygbdt.predict([1, 2, 1, 1]))
    print(mygbdt.score(DataSet[:, 0:4], DataSet[:, -1]))