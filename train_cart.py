

import numpy as np
# from DecisionTreeClass import BsTree
from DecisionTree.CartTree.CART_TREE_v_1_4 import BsTree
from DecisionTree.PlotTree.PlotTree_v_1_2 import PltTree
from Pretreat.dealData import dictmatrix
from Pretreat.cut import cut
import matplotlib.pyplot as plt
import numpy as np
import joblib

model_load_path = ""
model_save_path = "Decisionree/config/model1/"

train_file_path = "E:\\python\\project\\project2023\\ML_and_DL\\DecisionTree\\dataset\\set1/train_data.xlsx"
test_file_path = "E:\\python\\project\\project2023\\ML_and_DL\\DecisionTree\\dataset\\set1/text_data.xlsx"
vali_file_path = "E:\\python\\project\\project2023\\ML_and_DL\\DecisionTree\\dataset\\set1/verify_data.xlsx"




def train():
    dataset, label = dictmatrix(train_file_path)
    X_train = dataset[1:, :-1]
    Y_train = dataset[1:, -1]
    elemrowlist = range(X_train.shape[0])
    vail_data, _ = dictmatrix(vali_file_path)
    X_val = vail_data[1:, :-1]
    Y_val = vail_data[1:, -1]

    cart_model = None
    if model_load_path == "":
        # 重新生成树
        cart_model = BsTree(tree_class="regression", max_depth=50, min_sample_leaf=4, max_leaf_nodes=40, min_sample_split=6, min_impurity_decrease=0.005)
    elif model_load_path != "":
        cart_model = joblib.load(model_load_path)

    cart_model.fit(X_train, Y_train, elemrowlist, label, X_val, Y_val)

    joblib.dump(cart_model, "./config/cartmodel_1/cart4.dat")

    return

def test():

    # 加载模型
    test_data, lalist = dictmatrix(test_file_path)
    X_test = test_data[1:, :-1]
    Y_test = test_data[1:, -1]
    test_model = joblib.load("./config/cartmodel_1/cart4.dat")

    avg_score, y_pred = test_model.score(X_test, Y_test)
    draw_plot(Y_test, y_pred, avg_score)

def draw_plot(y1, y2, score):

    x = range(len(y1))
    plt.plot(x, y1, color='red', label='Real')
    plt.plot(x, y2, color='blue', label='Forecast')

    # 添加图例和标签
    plt.legend()
    plt.xlabel('X-Moives')
    plt.ylabel('Y-BoxOffice')
    plt.title('Average Score:'+str(score))

    # 显示图形
    plt.show()


if __name__ == "__main__":

    train()
    test()

    # testplot()
    print()



