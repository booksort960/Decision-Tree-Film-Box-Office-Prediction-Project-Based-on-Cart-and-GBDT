# Project Structure

## Introduce

This project is a movie box office prediction model trained and tested based on the movie dataset on Kaggle

For the data processing of the model, the model can only accept numeric matrices (int, double, float), but cannot accept strings in the matrix

When using cart for dataset partitioning, it is based on the row index of elements for partitioning. The selection of features is also based on the feature dictionary {"feature name": search and selection in the column index of the matrix}. The dataset in a tree only has one original copy and will not be added, deleted, or modified. We hope to reduce memory consumption by doing so

The cart in this project is a chained binary tree that implements tree nodes

Regarding pre pruning of cart, <font color=#ff9900 size=2.1 face="黑体">developer: I am not sure if cart actually has pre pruning technology. Different blogs have different opinions, and I cannot even determine which one is considered pre pruning. I am dizzy</font>. I set some parameters based on Skylearn's input, such as: maximum number of layers, maximum number of leaf nodes, and minimum number of split samples These parameters can be used to adjust the tree structure, which can effectively prevent overfitting

Regarding the post pruning CCP of cart, specific pruning was implemented, but cross validation with different alphas was not conducted
<font color=#ff9900 size=2.1 face="黑体">Developer: Lazy and unwilling to write, you can write a loop with a different alpha and call my post pruning function to obtain the optimal tree structure less than alpha, and then filter the optimal tree</font>

## Required Package Dependencies

        numpy,matplotlib,joblib,openpyxl,pandas

## File Instruction

    ConstructDecisionTree_ID3_dict.py
An ID3 decision tree with a dictionary tree structure

    train_cart.py
    train_gbdt.py

Used to read the model and start training

    cart_test.py
    gbdt_test.py

Used to test the functionality of cart and gbdt models

## Package functionality
    --CartTree
        Different versions of cart trees
    --GBDT
        Only one version of the GBDT model is retained
    --config
        Save some model files for gbdt and cart
    --dataset
        Save the original dataset used
    --Preteat
        It is a functional file for processing and slicing datasets into different files
    --PlotTree
        Visualization files for drawing decision trees in different versions

## If you have any questions or want to communicate, you can contact me

<font color=#ff9900 size=5 face="黑体">mail:2141166187@qq.com</font>