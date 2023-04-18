

from DecisionTree.GBDT.GBDT_v_1_2 import BsGBDT
from Pretreat.dealData import dictmatrix
from Pretreat.cut import cut
import matplotlib.pyplot as plt
import joblib

File_Path = "E:\\python\\project\\project2023\\ML_and_DL\\DecisionTree\\dataset\\lastoutputfile.xlsx"
Save_File_Path = "E:\\python\\project\\project2023\\ML_and_DL\\DecisionTree\\dataset\\set1"

model_load_path = ""
model_save_path = "Decisionree/config/model1/"

train_file_path = "E:\\python\\project\\project2023\\ML_and_DL\\DecisionTree\\dataset\\set1/train_data.xlsx"
test_file_path = "E:\\python\\project\\project2023\\ML_and_DL\\DecisionTree\\dataset\\set1/text_data.xlsx"
vali_file_path = "E:\\python\\project\\project2023\\ML_and_DL\\DecisionTree\\dataset\\set1/verify_data.xlsx"

def data_process():
    #
    cut(File_Path, Save_File_Path, 0.2, 0.1)


def train():
    dataset, label = dictmatrix(train_file_path)
    X_train = dataset[1:, :-1]
    Y_train = dataset[1:, -1]
    elemrowlist = range(X_train.shape[0])
    vail_data, _ = dictmatrix(vali_file_path)
    X_val = vail_data[1:, :-1]
    Y_val = vail_data[1:, -1]
    print(label)

    gbdt_model = None
    if model_load_path == "":
        # 重新生成树
        gbdt_model = BsGBDT(learning_rate=0.1,  n_estimators=101, subsample=1, max_depth=1000, min_sample_leaf=2, max_leaf_nodes=90, min_sample_split=2, min_impurity_decrease=0.00003)

    elif model_load_path != "":
        gbdt_model = joblib.load(model_load_path)

    gbdt_model.fit(X_train, Y_train, elemrowlist, label, X_val, Y_val, alpha_thres=0.000005)

    joblib.dump(gbdt_model, "./config/model_1/gbdt13.dat")
    return


def test():

    # 加载模型
    test_data, lalist = dictmatrix(test_file_path)
    X_test = test_data[1:, :-1]
    Y_test = test_data[1:, -1]
    test_model = joblib.load("./config/model_1/gbdt13.dat")

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

    #data_process()
    train()
    test()