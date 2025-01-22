import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iter=1000):
        """
        初始化感知机参数
        :param learning_rate: 学习率
        :param n_iter: 迭代次数
        """
        self.weights = None  # 权重初始化为 None
        self.bias = None  # 偏置初始化为 None

    def fit(self, X, y, iter=1000,learning_rate=0.01):
        """
        训练感知机
        :param X: 特征矩阵 (m, n)
        :param y: 标签 (m,)
        """
        m, n = X.shape  # m: 样本数量，n: 特征数量
        self.weights = np.zeros(n)  # 权重初始化为零
        self.bias = 0  # 偏置初始化为零
        y = np.array(y)
        # 迭代更新权重和偏置
        for _ in range(iter):
            for i in range(m):
                # 计算感知机的预测值
                linear_output = np.dot(X[i], self.weights) + self.bias
                predicted = self._activation(linear_output)

                # 计算梯度并更新权重和偏置
                error = y[i] - predicted
                self.weights += learning_rate * error * X[i]
                self.bias += learning_rate * error

    def predict(self, X):
        """
        用训练好的感知机进行预测
        :param X: 特征矩阵 (m, n)
        :return: 预测值 (m,)
        """
        linear_output = np.dot(X, self.weights) + self.bias
        return self._activation(linear_output)

    def _activation(self, x):
        """
        激活函数：单位阶跃函数
        :param x: 输入
        :return: 输出
        """
        return np.where(x >= 0, 1, 0)  # 若 x >= 0, 返回 1，否则返回 0
