import pandas as pd
import numpy as np
import GaussianMixtureModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 加载数据
data = pd.read_csv('support vector machine/Iris.csv')

# 删除id列（Iris数据集中没有id列，但为了统一格式，假设有一个id列）
data = data.drop(columns=['Id'])

# 将品种转换为数字标签（0：Iris-setosa，1：Iris-versicolor，2：Iris-virginica）
data['Species'] = data['Species'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})

# 将特征与目标变量分开
X = data.drop(columns=['Species'])
y = data['Species']

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分数据集，80%训练集，20%测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 输出数据集形状
print(f"训练集特征形状: {X_train.shape}")
print(f"测试集特征形状: {X_test.shape}")
print(f"训练集目标变量形状: {y_train.shape}")
print(f"测试集目标变量形状: {y_test.shape}")

# 训练
model = GaussianMixtureModel.GaussianMixture(n_components=3)
model.fit(X_train)

# 预测
y_pred = model.predict(X_test)

# 评估
accuracy = sum(y_pred == y_test) / len(y_test)
print("准确率: ", accuracy)



