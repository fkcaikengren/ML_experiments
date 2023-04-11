## 流程
    1.随机选择簇中心 
        随机种子
        
    2.计算每个样本到各个簇中心距离，划分簇
        基于欧氏距离

    3.重新计算簇中心
        针对每个簇内样本，计算特征均值向量作为簇中心

    4.迭代，重复步骤2,3
        每次迭代后更新inertia_和迭代计数器
        当inertia_不变化时或达到max_iter，停止迭代

    5.进行n_init次重复上述算法，选出最好的结果
        最好的结果评估标准是inertia_, inertia_越小越好

    ps. inertia_ : Sum of squared distances of samples to their closest cluster center

## K的选择

    1. 手肘法选取k值：绘制出k–inertia_图，看到有明显拐点的地方，设为k值，可以结合轮廓系数。
    2. 轮廓系数
        使用sklearn库的metrics.silhouette_score()方法可以很容易作出K—平均轮廓系数曲线。
        轮廓系数越大，K值越好。
    3. 结合业务理解，评估方法并不能一定选出最优的K值。