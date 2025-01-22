import numpy as np

class NaiveBayesian:
    def __init__(self):
        self.classes = None
        self.prior_prob = {}
        self.mean = {}
        self.var = {}

    def fit(self, x, y):
        # 获取唯一类别
        self.classes = np.unique(y)
        
        # 计算先验概率、均值和方差
        for cls in self.classes:
            x_c = x[y == cls]  # 按类别选择数据
            self.prior_prob[cls] = x_c.shape[0] / x.shape[0]  # 计算先验概率
            self.mean[cls] = np.mean(x_c, axis=0)  # 均值
            self.var[cls] = np.var(x_c, axis=0)  # 方差

    def predict(self, x):
        predictions = []
        
        # 对每个样本进行分类
        for sample in x:
            class_probs = {}
            
            # 计算每个类的 posterior probabilities
            for cls in self.classes:
                # 计算每个特征的likelihood
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
        probability_density = (1 / np.sqrt(2 * np.pi * var)) * \
                              np.exp(- (sample - mean) ** 2 / (2 * var))
        # 乘积
        return np.prod(probability_density)

