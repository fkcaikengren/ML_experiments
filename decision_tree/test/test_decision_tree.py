import os
from re import T
import sys

from sympy import plot
sys.path.append(os.getcwd())  # 相对路径：相对于文件test_*.py的路径

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from decision_tree import DescisionTree

# 加载数据
def load_data():
	data = np.array([
            [0, 0, 0, 0, 'no'],						
			[0, 0, 0, 1, 'no'],
			[0, 1, 0, 1, 'yes'],
			[0, 1, 1, 0, 'yes'],
			[0, 0, 0, 0, 'no'],
			[1, 0, 0, 0, 'no'],
			[1, 0, 0, 1, 'no'],
			[1, 1, 1, 1, 'yes'],
			[1, 0, 1, 2, 'yes'],
			[1, 0, 1, 2, 'yes'],
			[2, 0, 1, 2, 'yes'],
			[2, 0, 1, 1, 'yes'],
			[2, 1, 0, 1, 'yes'],
			[2, 1, 0, 2, 'yes'],
			[2, 0, 0, 0, 'no']
        ])
	feature_names = np.array(['F1-AGE', 'F2-WORK', 'F3-HOME', 'F4-LOAN'])		
	return data, feature_names

# 先根遍历树
def traverse_tree(tnode, level, visit):
    if not tnode:
        return
    
    visit(tnode, level)
    for i,sub_node in enumerate(tnode.feat_children.values()):
        traverse_tree(sub_node, level+1, visit)


# 树高
def get_tree_height(tnode):
    if not tnode.feat_children: #无子树
        return 1
    max_h = -1
    for sub_node in tnode.feat_children.values():
        h = get_tree_height(sub_node)
        max_h = max(max_h, h)

    return max_h+1

# 叶子节点个数
def get_leaf_cnt(tnode):
    cnt_dict = dict(cnt=0)
    def count_leaf(tnode,level): 
        if not tnode.feat_children:
            cnt_dict['cnt'] += 1
    traverse_tree(tnode, 0, count_leaf)
    return cnt_dict['cnt']



def plot_node(node_text, parent_pos, node_pos, edge_text, node_style=None):
    arrow_args = dict(arrowstyle="<-", connectionstyle="arc3")
    bbox_args = node_style or dict(boxstyle="Square,pad=0.4", fc="w")
    # s 标签； xy 注释的位置（就是箭头的起始位置）；xytext 标签的位置；
    # xycoords 注释位置（xy）坐标的类型；textcoords 标签位置(xytext)坐标的类型；
    plot_tree.ax.annotate(node_text, xy=parent_pos, xytext=node_pos, xycoords="data",textcoords="data",
            ha="center", va="center",
            bbox=bbox_args,  arrowprops=arrow_args )
    if edge_text is not None:
        plot_tree.ax.text((parent_pos[0]+node_pos[0])/2 , 
                        (parent_pos[1]+node_pos[1])/2, 
                        edge_text, va="center", ha="center")

def plot_nodes_recursive(node, parent_pos, level,width,  offset, edge_text=None):
    if not node:
        return

    text = feat_names[node.feat_index]  if node.feat_index else node.label
    text += "\n samples:"+str(node.num_sample) + "\n entropy:"+str(round(node.entropy,3))

    leaf_node = None
    if not node.feat_children: #是叶子
        leaf_node = plot_tree.leaf_node

    node_pos = ((offset+width/2)/plot_tree.width, 0.9-level/plot_tree.height)
    # print(node_pos)
    # 画当前结点  
    plot_node(text, 
              parent_pos or node_pos, 
              node_pos,
              edge_text,
              leaf_node)

    adjusted_node_pos = (node_pos[0], node_pos[1]-0.06)
    sub_offset = offset
    for feat_value,sub_node in node.feat_children.items():
        sub_width = get_leaf_cnt(sub_node)
        plot_nodes_recursive(sub_node, adjusted_node_pos, level+1, sub_width, sub_offset,feat_value)
        sub_offset += sub_width


def plot_tree(tnode):
    fig, ax = plt.subplots(figsize=(5,5))
    plot_tree.fig = fig
    plot_tree.ax = ax
    plot_tree.width = get_leaf_cnt(tnode)
    plot_tree.height = get_tree_height(tnode)


    plot_tree.leaf_node = dict(boxstyle="Square,pad=0.4", fc="lightgreen")
    plot_nodes_recursive(tnode, None, 0, plot_tree.width, 0, None)
    plt.show()






data,feat_names = load_data()
data_train = data[:, :-1]
labels = data[:, -1]
dtree = DescisionTree(data_train,labels)
dtree.train()
plot_tree(dtree.tree)

# 预测
res = dtree.predict(np.array([
    [1, 0, 0, 1],
    [2, 0, 1, 2],
]))
print(res)





