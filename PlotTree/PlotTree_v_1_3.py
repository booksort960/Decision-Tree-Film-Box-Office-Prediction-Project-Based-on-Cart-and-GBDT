import matplotlib.pyplot as plt

class PltTree:

    def __init__(self, intree):
        self._inTree = intree

    def _get_depth(self, node):
        if node is None:
            return 0
        return max(self._get_depth(node.Left), self._get_depth(node.Right)) + 1

    def _draw_node(self, node, x, y, width, height):
        if node.leaf == True:
            colors = "red"
        else:
            colors = "green"
        plt.gca().add_patch(plt.Rectangle((x-width/2, y-height/2), width, height, color=colors, alpha=0.3, fill=True))
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
        plt.text(x, y, str(sttr), ha='center', va='center', fontsize=5)

    def _draw_tree(self, node, x, y, width, height, depth, gap):
        if node is None:
            return
        self._draw_node(node, x, y, width, height)
        if node.Left is not None:
            self._draw_tree(node.Left, x-width/2-gap, y-height, width/2, height, depth-1, gap)
            plt.plot([x-gap/2, x-width/2+gap/2], [y-height/2+gap/2, y-gap/2], 'k')
        if node.Right is not None:
            self._draw_tree(node.Right, x+width/2+gap, y-height, width/2, height, depth-1, gap)
            plt.plot([x+gap/2, x+width/2-gap/2], [y-height/2+gap/2, y-gap/2], 'k')

    def _draw_binary_tree(self, root, gap=0.5):
        depth = self._get_depth(root)
        width = 2 ** (depth-1)
        height = 1
        self._draw_tree(root, width/2, height/2, width, height, depth, gap)
    def draw(self):
        root = self._inTree.get_root_node()
        self._draw_binary_tree(root, gap=0.5)
        plt.axis('off')
        plt.show()