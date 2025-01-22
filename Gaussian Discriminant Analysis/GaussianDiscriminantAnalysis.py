import numpy as np

class GaussianDiscriminator:
    def __init__(self):
        # 初始化类的属性
        self.classes = None
        self.prior_prob = {}
        self.mean = {}
        self.var = {}

    def fit(self, X, y):
        # 获取唯一类别
        self.classes = np.unique(y)
        
        # 计算每个类别的先验概率、均值和方差
        for cls in self.classes:
            X_c = X[y == cls]  # 按类别选择数据
            self.prior_prob[cls] = X_c.shape[0] / X.shape[0]  # 计算先验概率
            self.mean[cls] = np.mean(X_c, axis=0)  # 均值
            self.var[cls] = np.var(X_c, axis=0)  # 方差

    def predict(self, X):
        predictions = []
        
        for sample in X:
            class_probs = {}
            
            # 计算每个类的 posterior probabilities
            for cls in self.classes:
                # 计算每个类的 likelihood
                likelihood = self.calculate_likelihood(sample, cls)
                # 计算 posterior probability
                class_probs[cls] = self.prior_prob[cls] * likelihood
            
            # 选择概率最大的类
            predicted_class = max(class_probs, key=class_probs.get)
            predictions.append(predicted_class)

        return np.array(predictions)

    def calculate_likelihood(self, sample, cls):
        # 使用高斯分布计算似然概率
        mean = self.mean[cls]
        var = self.var[cls]

        # 为避免分母为零，使用一个很小的值（如 epsilon）
        epsilon = 1e-8
        var = np.where(var == 0, epsilon, var)

        probability_density = (1 / np.sqrt(2 * np.pi * var)) * \
                              np.exp(- (sample - mean) ** 2 / (2 * var))
        
        # 返回似然性，即乘积
        return np.prod(probability_density)
