'''
Name:
        绘制一棵决策树的类
Author:
        Booksort
Time:
        2023-4-15
Log:

Version:
        v0.1
Function:
        基于matplot绘制决策树
Think:

'''

from matplotlib import pyplot as plt
import numpy as np

class PltTree:
    def __init__(self, inTree, ax):
        self._inTree = inTree
        self.totalW = float(inTree.leaf_sample_sum)
        self.totalD = float(self._getTreeDepth(inTree.array_leaf)) * 2 + 1 # 包括节点与空隙需要多少层

        # 设置初始的x,y偏移量
        self.xOff = -0.5 / self.totalW
        self.yOff = 1.0
        self.ax = ax
        self.arrow_args = dict(arrowstyle="<-")

    def _getTreeDepth(self, leaf_list):
        '''
        遍历树中的所有叶子节点，获得整棵树的最大深度
        :param leaf_list: 树的叶子节点的列表
        :return: 树中最大树层数
        '''
        tree_node = max(leaf_list, key=lambda leaf_list: leaf_list.tree_depth)
        return tree_node.tree_depth

    def _get_same_depth_num(self, now_depth, node_list):
        sum = 1
        for node in node_list:
            if node.get_node_in_tree_depth() == now_depth:
                sum += 1

        return sum


    def _plot_mid_text(self, cntrPt, parentPt, txtString):
        xMid = (parentPt[0] + cntrPt[0]) / 2.0
        yMid = (parentPt[1] + cntrPt[1]) / 2.0

        # 计算 cntrPt、 parentPt 连线与水平方向的夹角
        if parentPt[0] - cntrPt[0] == 0:
            theta = 90
        else:
            theta = np.arctan((parentPt[1] - cntrPt[1]) / (parentPt[0] - cntrPt[0])) * 180 / np.pi

        self.ax.text(xMid, yMid, txtString, va="center", ha="center", rotation=theta)



    def _plot_node(self, nodeTxt, centerPt, parentPt, nodeType):
        #self.ax.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction', xytext=centerPt, textcoords='axes fraction', va="center", ha="center", arrowprops=self.arrow_args)
        plt.Rectangle()

    def _plot_tree(self, intree, parent_xy):
        '''
        层序遍历要绘制的树
        :param intree: 要绘制的树
        :param parent_xy: 初始节点的坐标
        :return:
        '''
        pyqueue = list()
        pyqueue.append(intree.get_root_node())
        xoff = self.xOff
        yoff = self.yOff
        while pyqueue.__len__() != 0:
            cur = pyqueue.pop(0)
            if cur.Left != None:
                pyqueue.append(cur.Left)
            if cur.Right != None:
                pyqueue.append(cur.Right)

            num_same_depth = self._get_same_depth_num(cur.tree_depth, intree.get_all_node_list())     # 获得与当前节点同一层的节点数量
            cntrPt = (self.xOff + (1.0 + float(num_same_depth)) / 2.0 / self.totalW, self.yOff)
            # self._plot_mid_text(cntrPt, parent_xy, cur.decValLabel)
            plt.pause(1)
            self._plot_node(cur.decValLabel, cntrPt, parent_xy, type(cur))
            plt.pause(1)
            # xoff = xoff + 1.0 / self.totalW
            if cur.parent != None and cur.parent.Right == cur:
                yoff = yoff - 1.0 / self.totalD
            parent_xy = (xoff, yoff)


        return

    def draw(self):
        self._plot_tree(self._inTree, (0.5, 1))
        plt.show()
if __name__ == "__main__":

    print()
