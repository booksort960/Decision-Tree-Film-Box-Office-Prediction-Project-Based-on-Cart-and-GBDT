from matplotlib import pyplot as plt
from collections import deque


def draw_tree(root):
    if not root:
        return
    # 设置节点大小和间距
    node_size = 40
    node_space = 20
    # 创建画布和子图
    fig, ax = plt.subplots(figsize=(10, 10))
    # 设置坐标轴范围
    ax.set_xlim(0, 2**(get_depth(root)+1)*node_size)
    ax.set_ylim(0, (get_depth(root)+1)*node_size + get_depth(root)*node_space)

    # 创建队列，用于层次遍历二叉树
    queue = deque([(root, 2**get_depth(root)/2*node_size, node_size/2)])

    # 遍历队列，绘制节点和连线
    while queue:
        node, x, y = queue.popleft()

        # 绘制节点矩形
        rect = plt.Rectangle((x-node_size/2, y-node_size/2), node_size, node_size, fill=False)
        ax.add_patch(rect)

        # 绘制节点值
        plt.text(x, y, str(node.tree_depth), ha='center', va='center')

        # 绘制左子树
        if node.Left:
            queue.append((node.Left, x-2**(get_depth(node.Left)-1)*node_size, y+node_size+node_space))

            # 绘制左子树连线
            plt.plot([x, x-2**(get_depth(node.Left)-1)*node_size], [y+node_size/2, y+node_size+node_space-node_size/2], 'k-')

        # 绘制右子树
        if node.Right:
            queue.append((node.Right, x+2**(get_depth(node.Right)-1)*node_size, y+node_size+node_space))

            # 绘制右子树连线
            plt.plot([x, x+2**(get_depth(node.Right)-1)*node_size], [y+node_size/2, y+node_size+node_space-node_size/2], 'k-')

    # 显示图形
    plt.axis('off')
    plt.show()

def get_depth(root):
    if not root:
        return 0
    return max(get_depth(root.Left), get_depth(root.Right)) + 1