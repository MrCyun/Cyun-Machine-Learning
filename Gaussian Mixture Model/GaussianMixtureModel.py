import numpy as np
from scipy.stats import multivariate_normal

class GaussianMixture:
    def __init__(self, n_components, max_iter=10000, tol=1e-3):
        self.n_components = n_components  # 混合成分个数
        self.max_iter = max_iter          # 最大迭代次数
        self.tol = tol                    # 收敛阈值
        self.means = None                 # 高斯成分均值
        self.covariances = None           # 高斯成分协方差
        self.weights = None                # 每个成分的权重

    def fit(self, X):
        n_samples, n_features = X.shape
        
        # 初始化
        self.weights = np.ones(self.n_components) / self.n_components
        self.means = X[np.random.choice(n_samples, self.n_components, False)]
        self.covariances = np.array([np.eye(n_features)] * self.n_components)

        log_likelihoods = []

        for _ in range(self.max_iter):
            # E步：计算责任分配
            responsibilities = self.e_step(X)

            # M步：更新参数
            self.m_step(X, responsibilities)

            # 计算对数似然
            log_likelihood = self.calculate_log_likelihood(X)
            log_likelihoods.append(log_likelihood)

            # 检查收敛
            if len(log_likelihoods) > 1 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < self.tol:
                break

    def e_step(self, X):
        likelihoods = np.array([
            multivariate_normal.pdf(X, mean=self.means[i], cov=self.covariances[i])
            for i in range(self.n_components)
        ])
        weighted_likelihoods = self.weights[:, np.newaxis] * likelihoods
        responsibilities = weighted_likelihoods / weighted_likelihoods.sum(axis=0)
        return responsibilities

    def m_step(self, X, responsibilities):
        total_responsibility = responsibilities.sum(axis=1)

        # 更新权重
        self.weights = total_responsibility / total_responsibility.sum()

        # 更新均值
        self.means = np.array([
            (responsibilities[i][:, np.newaxis] * X).sum(axis=0) / total_responsibility[i]
            for i in range(self.n_components)
        ])
        
        # 更新协方差
        self.covariances = np.array([
            (responsibilities[i][:, np.newaxis] * (X - self.means[i])).T @ (X - self.means[i]) / total_responsibility[i]
            for i in range(self.n_components)
        ])

    def calculate_log_likelihood(self, X):
        likelihoods = np.array([
            multivariate_normal.pdf(X, mean=self.means[i], cov=self.covariances[i])
            for i in range(self.n_components)
        ])
        weighted_likelihoods = self.weights[:, np.newaxis] * likelihoods
        return np.sum(weighted_likelihoods)

    def predict(self, X):
        responsibilities = self.e_step(X)
        return np.argmax(responsibilities, axis=0)
