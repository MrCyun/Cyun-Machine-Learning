import numpy as np

class Kmeans:
    def __init__(self, k, max_iter=100, tol=1e-4):
        self.k = k  # 簇的数量
        self.max_iter = max_iter  # 最大迭代次数
        self.tol = tol  # 收敛阈值
        self.centers = None  # 簇心
        self.labels = None  # 每个样本所属的簇标签

    def fit(self, X):
        n_samples, n_features = X.shape
        
        # 随机初始化簇心
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centers = X[random_indices]

        for _ in range(self.max_iter):
            # 计算每个点到簇心的距离
            distances = self.compute_distances(X)
            # 将每个点分配给最近的簇心
            new_labels = np.argmin(distances, axis=1)

            # 检查是否收敛
            if np.all(new_labels == self.labels):
                break

            self.labels = new_labels
            
            # 更新簇心
            self.update_centers(X)

    def compute_distances(self, X):
        # 计算每个样本到每个簇心的欧几里得距离
        distances = np.linalg.norm(X[:, np.newaxis] - self.centers, axis=2)
        return distances

    def update_centers(self, X):
        for i in range(self.k):
            if np.any(self.labels == i):  # 防止选择空簇
                self.centers[i] = X[self.labels == i].mean(axis=0)

    def predict(self, X):
        # 使用已经训练好的模型进行预测
        distances = self.compute_distances(X)
        return np.argmin(distances, axis=1)  # 返回每个样本的簇标签

    def get_centers(self):
        return self.centers