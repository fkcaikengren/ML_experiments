import os
import sys

sys.path.append(os.getcwd())  # 相对路径：相对于test_bar.py的路径

from logistic_regression import LogisticRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('./logistic_regression/data/microchips-tests.csv')

x_axis = 'param_1'
y_axis = 'param_2'

label_name = 'validity'

data_train = data[[x_axis, y_axis]].values
labels = data[label_name].values
unique_labels = np.unique(labels)




def paint_scatter():
    for u_label in unique_labels:
        x = data[x_axis][labels==u_label]
        y = data[y_axis][labels==u_label]
        plt.scatter(x, y, label=u_label)
        plt.legend()
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.show()


def paint_loss(cost_history):
    for i,cost in enumerate(cost_history):
        plt.plot( np.arange(len(cost)), cost, label=unique_labels[i])
    plt.xlabel('iteration count')
    plt.ylabel('cost')
    plt.show()

def paint_boundary(lr):
    num_samples = data.shape[0]
    x_min = np.min(data[x_axis])
    x_max = np.max(data[x_axis])*1.5
    y_min = np.min(data[y_axis])
    y_max = np.max(data[y_axis])*1.5

    for type in unique_labels: 
        plt.scatter(
            data[x_axis][data[label_name]==type], 
            data[y_axis][data[label_name]==type],
            label=type
        )
    x = np.linspace(x_min, x_max, num_samples)
    y = np.linspace(y_min, y_max, num_samples)
    X,Y = np.meshgrid(x, y)
   
    Z = np.zeros((num_samples, num_samples))
    # print(X, Y)
    for idx, y_row in enumerate(Y):
        # X,Y的行
        x_row = X[idx]
        data_row = np.zeros((num_samples,2))
        data_row[:,0] = x_row
        data_row[:,1] = y_row
        class_prediction = lr.predict(data_row)
        Z[idx] = class_prediction
    # print(Z[:,30:60])
    contours = plt.contour(X,Y,Z) #绘制等高线（轮廓线）
    plt.show()

def test_accuracy(lr):
    class_index = lr.predict(data_train)
    class_prediction = np.empty(class_index.shape, dtype=object)
    for index,u_label in enumerate(unique_labels):
        class_prediction[class_index == index] = u_label #筛选赋值
    accuracy = (1-class_prediction[class_prediction != labels].shape[0]/labels.shape[0])*100
    print(f"accuracy of train data: {accuracy}%")




lr = LogisticRegression(data_train, labels, 4)
cost_history = lr.train()

print('最佳参数：')
print(lr.theta)

# Loss 迭代图
paint_loss(cost_history)

# 训练集的准确率
test_accuracy(lr)

# 决策边界
paint_boundary(lr)


