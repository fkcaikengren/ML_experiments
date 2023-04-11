import os
import sys

sys.path.append(os.getcwd())  # 相对路径：相对于test_bar.py的路径

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from logistic_regression import LogisticRegression
from logistic_regression.utils.features import normalize

"""
测试标准化的效果：从准确率和迭代次数两方面
"""

x_axis = 'sepal_length'
y_axis = 'petal_width'


data = pd.read_csv('./logistic_regression/data/iris.csv')
data_train = data[[x_axis, y_axis]].values 
labels = data['class'].values 
unique_labels = np.unique(labels)


def paint_scatter(data):
    for u_label in unique_labels:
        x = data[:,0][labels==u_label]
        y = data[:,1][labels==u_label]
        plt.scatter(x, y, label=u_label)
        plt.legend()
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.show()

def test_accuracy(lr):
    class_index = lr.predict(data_train)
    class_prediction = np.empty(class_index.shape, dtype=object)
    for index,u_label in enumerate(unique_labels):
        class_prediction[class_index == index] = u_label #筛选赋值
    accuracy = (1-class_prediction[class_prediction != labels].shape[0]/labels.shape[0])*100
    print(f"accuracy of train data: {accuracy}%")


# 样本的值很大时，导致无法收敛
def test_bad_data():
    data_train = np.array([
        [510,20,],
        [700,140],
        [590,180],
    ])
    labels = np.array(['SETOSA', 'VERSICOLOR', 'VIRGINICA'])
    lr = LogisticRegression(data_train, labels,0,False) 
    lr.train()
    # 当学习率alpha=0.1，初始theta=[0,0], 采用sigmoid激活函数，上述代码报错
    # 原因分析：训练时无法收敛
    # sigmoid(40) = 1.0, 当假设函数值x>=40时，将导致sigmoid恒为1，
    # 由于样本的特征值比较大，加上alpha比较大，梯度下降过多，导致损失函数的迭代值不收敛反而越来越大！
    # 解决：1）将alpha从0.1降至 0.00001； 2）标准化样本数据，将值大幅度缩小


# 未标准化的数据
# paint_scatter(data_train)
# 标准化的数据
# data_train_normalized,_mean,_std = normalize(data_train)
# paint_scatter(data_train_normalized)




lr = LogisticRegression(data_train, labels, 0, False)
# 迭代次数为：21,22,27,
lr.train()
test_accuracy(lr)
lr2 = LogisticRegression(data_train, labels,0,True) 
# 迭代次数为：13,7,14,
lr2.train()
test_accuracy(lr2)

# 标准化后的迭代次数少很多！
