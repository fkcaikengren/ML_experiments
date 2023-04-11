
## Issue

使用scipy的minimize 遇到：Desired error not necessarily achieved due to precision loss.
    1.直接原因是梯度下降无法收敛，可能是方向反了/学习率过大
    2.间接原因是cost function没写对



## 数据预处理：
    1.标准化
        好处：1）加快梯度下降速度（当学习率alpha偏小时）
             2）有助于收敛，提升模型参数精确度（当学习率alpha偏大时）
    2.多项式（维度提升）