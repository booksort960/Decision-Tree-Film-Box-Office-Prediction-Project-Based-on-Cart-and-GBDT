from matplotlib import pyplot as plt

class PltTree:

    def __init__(self, intree):
        self._inTree = intree

    def _draw_node(self, node, x, y, width, height):
        if node.leaf == True:
            colors = "red"
        else:
            colors = "green"
        plt.gca().add_patch(plt.Rectangle((x - width/2, y - height/2), width, height, color=colors, alpha=0.3, fill=True))

        if self._inTree.get_tree_class() == "classify":
            messtr = "gini=" + str(round(node.gini, 2)) + '\n' \
                     + "samples=" + str(len(node.beforevalueIndex)) + '\n' \
                     + "class=" + str(node.classresult)

        else:
            messtr = "squared_err=" + str(round(node.errsquared, 2)) + '\n' \
                     + "samples=" + str(len(node.beforevalueIndex)) + '\n' \
                     + "value=" + str(node.classresult)
        if node.leaf == False:
            sttr = node.decValLabel + "<=" + str(node.threshold) + '\n' + messtr
        else:
            sttr = messtr
        plt.text(x, y, str(sttr), ha='center', va='center', fontsize=7)

    # 定义绘制二叉树的函数
    def _draw_tree(self, root, x, y, width, height):
        if root is None:
            return
        self._draw_node(root, x, y, width, height)
        if root.Left is not None:
            plt.plot([x, x - width*1.5], [y - height/2, y - height*3/2], 'k-')
            self._draw_tree(root.Left, x - width*1.5, y - height*3/2, width, height)
        if root.Right is not None:
            plt.plot([x, x + width*1.5], [y - height/2, y - height*3/2], 'k-')
            self._draw_tree(root.Right, x + width*1.5, y - height*3/2, width, height)

    # 设置画布大小和坐标轴范围

    def draw(self):
        root = self._inTree.get_root_node()
        self._draw_tree(root, 0, 0, 1.5, 1)
        plt.figure(figsize=(10, 10))
        plt.axis('equal')
        plt.axis('off')
        plt.show()
    # 绘制二叉树
    #draw_tree(root, 0, 0, 1.5, 1)

    # 显示图形
