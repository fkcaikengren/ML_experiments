import numpy as np
from scipy.optimize import minimize
from .utils.features import preprocess_data
from .utils.hypothesis import sigmoid

class LogisticRegression:
    def __init__(self, data, labels, polynomial_degree=0, normalize_data=False) :
        self.labels = labels
        self.polynomial_degree = polynomial_degree
        self.normalize_data = normalize_data

        data_processed,features_mean,features_std_deviation = preprocess_data(data, polynomial_degree, normalize_data)
        self.data = data_processed
        self.features_mean = features_mean
        self.features_std_deviation = features_std_deviation
        
        self.unique_labels = np.unique(labels)
        num_features = self.data.shape[1]
        num_unique_labels = self.unique_labels.shape[0]
        self.theta = np.zeros((num_unique_labels, num_features)) # 多分类问题：各个分类各有一组参数
        
    def train(self, max_iter=1000):
        cost_history = []
        for index, unique_label in enumerate(self.unique_labels):
            # label数值化
            cur_labels = np.array(
                [ 1 if label==unique_label else 0 for label in self.labels]
                )
            optimised_theta,cost_hist = LogisticRegression.gradient_descent(
                self.data, 
                cur_labels, 
                self.theta[index], 
                max_iter)
            cost_history.append(cost_hist)
            self.theta[index] = optimised_theta
        
        return cost_history

    def predict(self, data):
        data_processed,features_mean,features_std_deviation = preprocess_data(data, self.polynomial_degree, self.normalize_data)
        # 得到概率矩阵：行数为数据集数量，列数为分类数量
        prob_matrix = LogisticRegression.hypothesis(data_processed, self.theta.T) 
        # x轴方向最大的值索引
        class_index = np.argmax(prob_matrix,axis=1)  
        # print(f"class_index: {class_index}")
        
        return class_index

    @staticmethod
    def gradient_descent(data, labels, init_theta, max_iter):
        """
            data: 二维数组
            labels：一维数组 (二值化的一维数组，例如：[0,1,0,1,1])
            init_theta: 一维数组
            max_iter: 数值，最大迭代次数
            返回值：（最优参数，历史损失值）
        """
        # print(LogisticRegression.cost_function(data, labels, np.array([ 0,0,0])))

        cost_func = lambda cur_theta : LogisticRegression.cost_function(data, labels, cur_theta)
        derivative_func = lambda cur_theta: LogisticRegression.gradient_step(data, labels, cur_theta)
        cost_history = []
        res = minimize(
                cost_func,
                init_theta,
                method="CG",
                jac= derivative_func,
                options={ "disp":True, "maxiter": max_iter},
                callback = lambda cur_theta:cost_history.append(
                    LogisticRegression.cost_function(data, labels, cur_theta)
                    )
                )
        if not res.success:
            print(res)
            raise ArithmeticError('Can not minimize cost function: '+res.message)
        return (res.x, cost_history)
    
    @staticmethod
    def gradient_step(data, labels, theta):
        """
            data: 二维数组
            labels：一维数组
            theta: 一维数组
            返回值：导数值向量（维度大小为特征总数）
        """
        alpha = 0.1
        m = data.shape[0] # 样本总数
        f_theta = LogisticRegression.hypothesis(data, theta) 
        tmp = (alpha/m) * np.dot((f_theta - labels), data) 
        # print(f"---tmp.shape: {tmp.shape}") # shape is (3,)
        # print(tmp)
        return tmp

    @staticmethod
    def cost_function(data, labels, theta):
        """
            data: 二维数组
            labels：一维数组
            theta: 一维数组
            返回值：损失函数值
        """
        m = data.shape[0] # 样本总数
        f_theta = LogisticRegression.hypothesis(data, theta)
        cost = (-1/m) * (
            np.dot(labels[labels==1], np.log(f_theta[labels==1])) +
            np.dot(1-labels[labels==0], np.log(1-f_theta[labels==0]))
        )
        return cost

    @staticmethod
    def hypothesis(data, theta):
        """
            data: 二维数组
            theta: 一维数组/二维数组
            返回值：假设函数值向量，维度大小为样本总数
        """
        z = np.dot(data, theta) # z向量
        f = sigmoid(z) # f向量
        return f
    
