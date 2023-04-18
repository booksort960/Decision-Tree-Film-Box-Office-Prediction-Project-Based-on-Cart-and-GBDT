'''
Name:
        绘制一棵决策树的类
Author:
        Booksort
Time:
        2023-4-15
Log:

Version:
        v0.2
Function:
        基于matplot绘制决策树
Think:

'''

from matplotlib import pyplot as plt
import numpy as np

class PltTree:
    def __init__(self, inTree):
        self._inTree = inTree
        self.totalW = float(self._getTreeWidth(inTree))
        self.totalD = float(self._getTreeDepth(inTree.array_leaf)) # 包括节点与空隙需要多少层

        # 设置初始的x,y偏移量
        self.xOff = -0.5 / self.totalW
        self.yOff = 1.0

        self.arrow_args = dict(arrowstyle="<-")

        # 节点自身数据
        self.d_hor = 4  # 节点水平距离
        self.d_vec = 8  # 节点垂直距离


    def _getTreeWidth(self, intree):
        leftnum = 0
        rightnum = 0
        root = intree.get_root_node()
        cur = root
        while cur.Left != None:
            cur = cur.Left
            leftnum += 1
        cur = root
        while cur.Right != None:
            cur = cur.Right
            rightnum += 1

        return rightnum + leftnum + 1

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


    def _plot_mid_line(self, x1, y1, x2, y2):
        x = (x1, x2)
        y = (y1, y2)
        plt.plot(x, y, 'k-')



    def _plot_node(self, x, y, node, ax):
        if node.leaf == True:
            colors = "red"

        else:
            colors = "green"
        c_node = plt.Rectangle(
            (x-self.d_vec/2, y-self.d_hor),  # (x,y)矩形左下角
            self.d_vec,  # width长
            self.d_hor,  # height宽
            color=colors, alpha=0.3)


        # c_node = plt.Circle((x, y), radius=self.radius, color='green')
        ax.add_patch(c_node)
        if self._inTree.get_tree_class() == "classify":
            messtr = "gini=" + str(round(node.gini, 2)) + '\n' \
                + "samples=" + str(len(node.beforevalueIndex))+ '\n' \
                + "class=" + str(node.classresult)

        elif self._inTree.get_tree_class() == "regression":
            messtr = "squared_err=" + str(round(node.errsquared, 2)) + '\n' \
                + "samples=" + str(len(node.beforevalueIndex)) + '\n' \
                + "value=" + str(node.classresult)


        if node.leaf == False:
            sttr = node.decValLabel + "<=" + str(node.threshold) + '\n' + messtr
            plt.text(x, y-self.d_hor, '%s' % sttr, ha='center', va='bottom', fontsize=5)
        else:
            sttr = messtr
            plt.text(x, y-self.d_hor, '%s' % sttr, ha='center', va='bottom', fontsize=5)



    def _plot_tree(self, x, y, ax):
        '''
        层序遍历要绘制的树
        :param intree: 要绘制的树
        :param parent_xy: 初始节点的坐标
        :return:
        '''
        root = self._inTree.get_root_node()

        pyqueue = list()
        xylist = list()
        pyqueue.append(root)
        xylist.append((x, y))
        lx = rx = 0
        ly = ry = y - self.d_vec
        while pyqueue.__len__() != 0:
            cur = pyqueue.pop(0)
            x, y = xylist.pop(0)
            self._plot_node(x, y, cur, ax)
            lx = rx = 0
            ly = ry = y - self.d_vec
            if cur.Left != None:
                pyqueue.append(cur.Left)
                lcur = cur.Left
                lsum = 0
                while lcur.Right != None:
                    lcur = lcur.Right
                    lsum += 1
                lx = x - self.d_hor * (lsum + 1.5)
                self._plot_mid_line(x, y-self.d_hor, lx, ly)
                xylist.append((lx, ly))
            if cur.Right != None:
                pyqueue.append(cur.Right)
                rcur = cur.Right
                rsum = 0
                while rcur.Left != None:
                    rcur = rcur.Left
                    rsum += 1
                rx = x + self.d_hor * (rsum + 1.5)  # x-右子树的左边宽度
                self._plot_mid_line(x, y-self.d_hor, rx, ry)
                xylist.append((rx, ry))

        return


    def _create_win(self, d_hor, d_vec):
        WEIGHT = self.totalW
        HEIGHT = self.totalD
        WEIGHT = (WEIGHT + 1) * d_hor
        HEIGHT = (HEIGHT + 1) * d_vec
        fig = plt.figure(figsize=(11, 9))
        ax = fig.add_subplot(111)
        plt.xlim(0, WEIGHT*5)
        plt.ylim(0, HEIGHT*2)
        lnum = 0
        cur = self._inTree.get_root_node()
        while cur.Left != None:
            cur = cur.Left
            lnum += 1
        x = (lnum + 3) *3* d_hor  # x, y 是第一个要绘制的节点坐标，由其左子树宽度决定
        y = 1.2 * HEIGHT - d_vec
        return fig, ax, x, y


    def draw(self):
        _, ax, x, y = self._create_win(self.d_hor, self.d_vec)
        self._plot_tree(x, y, ax)
        plt.show()
if __name__ == "__main__":

    print()
