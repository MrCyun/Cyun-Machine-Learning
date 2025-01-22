import numpy as np

class linear_regression:
    def __init__(self):
        self.theta = None  # 模型参数
        self.X = None  # 特征
        self.y = None  # 目标值

    def fit(self, X, y, iters=1000, alpha=0.01):
        # 确保 X 和 y 的维度正确
        m = len(y)  # 样本数量
        n = X.shape[1]  # 特征数量
        self.X = X
        self.y = y.values.reshape(-1, 1)  # 将 y 转换为列向量 (m, 1)
        # 初始化 theta 为零
        self.theta = np.zeros((n, 1))
        # 执行梯度下降
        for i in range(iters):
            # 计算预测值 hx
            hx = np.dot(X, self.theta)
            # 计算损失函数 (hx - y)
            loss = hx - self.y
            # 计算梯度
            gradient = (2/m) * np.dot(X.T, loss)
            # 更新 theta
            self.theta -= alpha * gradient

        return self.theta  # 返回训练得到的参数

    def predict(self, X):
        return np.dot(X, self.theta)

    def get_params(self):
        return self.theta  # 获取模型参数 theta

