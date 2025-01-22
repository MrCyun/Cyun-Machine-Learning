import numpy as np
from scipy.optimize import minimize

class SVM:
    def __init__(self, C=1.0):
        self.C = C  # C 控制软间隔的惩罚项（对于线性可分数据，C 可以设为一个较大的值）

    def fit(self, X, y):
        self.models = []  # 存储每个类别的模型
        
        # 针对每个类别进行训练（采用一对多策略）
        classes = np.unique(y)
        
        for class_label in classes:
            # 创建二分类标签，将当前类别作为正类，其他类别作为负类
            binary_y = np.where(y == class_label, 1, -1)
            
            # 使用当前类别的标签训练 SVM 模型
            model = self._train_binary_svm(X, binary_y)
            self.models.append(model)

    def _train_binary_svm(self, X, y):
        m, n = X.shape  # 获取样本数和特征数
        y = np.where(y <= 0, -1, 1)  # 确保标签是 -1 或 1

        # 初始化对偶问题中的拉格朗日乘子 alpha
        alpha = np.zeros(m)
        
        # 使用 scipy 的 minimize 函数求解对偶问题
        def objective(alpha):
            # 计算目标函数 W(α) = Σα_i - 1/2 Σα_i α_j y_i y_j (x_i · x_j)
            # 这是 SVM 的对偶问题的目标函数
            alpha = alpha.reshape(-1, 1)
            kernel = np.dot(X, X.T)  # 线性核函数，计算点积
            return 0.5 * np.sum(np.outer(alpha, alpha) * kernel * np.outer(y, y)) - np.sum(alpha)

        # 约束条件：Σα_i y_i = 0 (这是拉格朗日对偶问题的约束)
        def constraint(alpha):
            return np.dot(alpha, y)

        # 确保 α_i >= 0
        bounds = [(0, self.C) for _ in range(m)]

        # 约束和目标函数传入 minimize 求解
        constraints = [{'type': 'eq', 'fun': constraint}]
        result = minimize(objective, alpha, bounds=bounds, constraints=constraints, method='SLSQP')

        # 取得最优解
        alpha = result.x

        # 计算 w 和 b
        w = np.sum(alpha[:, None] * y[:, None] * X, axis=0)
        support_vectors_indices = alpha > 1e-5
        b = np.mean(y[support_vectors_indices] - np.dot(X[support_vectors_indices], w))

        return (w, b, alpha)

    def predict(self, X_test):
        # 对于每个分类器，计算其预测分数
        scores = np.array([np.dot(X_test, model[0]) + model[1] for model in self.models])
        
        # 获取得分最高的类别
        predictions = np.argmax(scores, axis=0)
        
        return predictions
