import numpy as np


"""
    功能：标准化
        x = (x-mean(x))/standard_deviation(x)
"""
def normalize(data):
    data_processed = data.astype(float,copy=True)
    
    features_mean = np.mean(data_processed, axis=0)
    features_std_deviation = np.std(data_processed, axis=0, ddof=1)
    
    for index,row in enumerate(data_processed):
        data_processed[index] = (row-features_mean)/features_std_deviation

    return data_processed,features_mean,features_std_deviation

"""
    功能：数据升维（多项式变换）
        将两个特征升维，例如polynomial_degree=3时，
        [x1, x2] => [1, x1, x2, x1^2, x2^2, x1*x2, x1^3, x1^2*x2, x1*x2^2, x2^3].
"""
def polynomial_of_2features(data, polynomial_degree):

    polynomials = np.empty((data.shape[0], 0))
    feature_1 = data[:, 0]
    feature_2 = data[:, 1] #一维数组
    
    for i in range(polynomial_degree+1):
        for j in range(i+1):
            polynomial_features = np.power(feature_1, i-j) * np.power(feature_2, j)
            polynomial_features = polynomial_features.reshape(-1,1) # 转为二维数组
            polynomials = np.concatenate((polynomials, polynomial_features), axis=1) #水平拼接
    
    return polynomials


"""
    返回值：(预处理数据，均值，标准差)
"""
def preprocess_data(data, polynomial_degree=0, normalize_data=True):
    num_samples = data.shape[0] # 样本总数
    num_features = data.shape[1]
    data_processed = np.copy(data)

    # 期望
    features_mean = 0
    # 方差
    features_std_deviation = 0

    # 标准化
    if normalize_data:
        data_processed,features_mean,features_std_deviation = normalize(data_processed)

    # 多项式（维度提升）
    if polynomial_degree>0 and num_features==2: #简单起见，只支持两个特征的多项式
        # 第一列为 1
        data_processed = polynomial_of_2features(data_processed, polynomial_degree)
    else :
        # 加上一列 1
        data_processed = np.hstack((np.ones((num_samples,1)), data_processed))

    return data_processed,features_mean,features_std_deviation