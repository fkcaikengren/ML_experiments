import os
import sys
sys.path.append(os.getcwd())  # 相对路径：相对于文件test_*.py的路径

from logistic_regression import LogisticRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

    
# np.set_printoptions(threshold=np.inf) #无限制打印np.ndarray

x_axis = 'petal_length'
y_axis = 'petal_width'
iris_types = np.array(['SETOSA', 'VERSICOLOR', 'VIRGINICA'])

data = pd.read_csv('./logistic_regression/data/iris.csv')
data_train = data[[x_axis, y_axis]].values 
labels = data['class'].values 

def test_pd():
    # 相对运行目录
    print(data.index)
    print(data.columns) 
    print(data.values) #二维数组
    print(data['class']) #series对象
    print(data['class'].values)  #一维数组

def paint_scatter():
    # 画数据的散列图
    for type in iris_types: 
        plt.scatter(
            #data[x_axis]是列series, data[x_axis][data['class']==type]是筛选后的series
            data[x_axis][data['class']==type], 
            data[y_axis][data['class']==type],
            label=type
        )
    plt.show()

def paint_loss():
    for i,cost in enumerate(cost_history):
        plt.plot(np.arange(len(cost)), cost, label=iris_types[i])
        plt.xlabel('iteration count')
        plt.ylabel('cost')
    plt.show()


def paint_boundary(lr):
    num_samples = data.shape[0]
    x_min = np.min(data[x_axis])
    x_max = np.max(data[x_axis])*1.5
    y_min = np.min(data[y_axis])
    y_max = np.max(data[y_axis])*1.5

    for type in iris_types: 
        plt.scatter(
            data[x_axis][data['class']==type], 
            data[y_axis][data['class']==type],
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
    plt.clabel(contours, inline=True, fontsize=8) #显示等高线值
    plt.show()



   



 # Loss收敛的折线图


def test_accuracy(lr):
    class_index = lr.predict(data_train)
    class_prediction = np.empty(class_index.shape, dtype=object)
    for index,u_label in enumerate(iris_types):
        class_prediction[class_index == index] = u_label #筛选赋值
    accuracy = (1-class_prediction[class_prediction != labels].shape[0]/labels.shape[0])*100
    print(f"accuracy of train data: {accuracy}%")




lr = LogisticRegression(data_train, labels)
cost_history = lr.train()

# 画散点图
paint_scatter()

print('最佳参数：')
print(lr.theta)

# Loss迭代图
paint_loss()

# 训练集的准确率
test_accuracy(lr)

# 画决策边界
paint_boundary(lr)