import numpy as np

class LogisticRegression:
    def __init__(self):
        self.theta = None

    def fit(self, X, y, iters=1000, alpha=0.01):
        """
        训练模型
        :param X: 训练集特征矩阵 (m, n)
        :param y: 训练集标签 (m,)
        :param iters: 迭代次数
        :param alpha: 学习率
        """
        m, n = X.shape  # 样本数量 m，特征数量 n
        self.theta = np.zeros((n, 1))  # 初始化 theta 为零
        y = y.values.reshape(-1, 1)  # 确保 y 是列向量

        # 梯度下降
        for i in range(iters):
            h = self._sigmoid(np.dot(X, self.theta))  # 计算预测值
            gradient = np.dot(X.T, h - y) / m  # 梯度计算
            # 更新 theta
            self.theta -= alpha * gradient


    def predict(self, X, probability=False):
        """
        预测函数
        :param X: 特征矩阵 (m, n)
        :param probability: 是否返回概率值，默认返回类别（0 或 1）
        :return: 预测结果（类别或概率）
        """
        prob = self._sigmoid(np.dot(X, self.theta))  # 预测概率
        if probability:
            return prob  # 返回概率值
        return (prob >= 0.5).astype(int)  # 根据阈值 0.5 转换为 0 或 1

    @staticmethod
    def _sigmoid(z):
        """Sigmoid 激活函数"""
        return 1 / (1 + np.exp(-z))

    def _compute_loss(self, X, y):
        """
        计算损失函数
        :param X: 特征矩阵 (m, n)
        :param y: 标签 (m,)
        :return: 损失值
        """
        m = len(y)
        h = self._sigmoid(np.dot(X, self.theta))  # 计算预测值
        loss = (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()  # 二元交叉熵损失函数
        return loss
