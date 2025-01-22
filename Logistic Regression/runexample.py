import pandas as pd
import numpy as np
import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
data = pd.read_csv('logistic regression/breast-cancer.csv')
# 删除id列
data = data.drop(columns=['id'])
# 将诊断结果转换为0和1
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
# 将特征与目标变量分开
X = data.drop(columns=['diagnosis'])
y = data['diagnosis']
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
model = LogisticRegression.LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# 评估
y_pred = y_pred.flatten()
# accuracy
accuracy = np.sum(y_pred == y_test) / len(y_test)
print("Accuracy: ", accuracy)

# 对比sklearn的结果
print("对比sklearn的结果")
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0).fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = np.sum(y_pred == y_test) / len(y_test)
print("Sklearn Accuracy: ", accuracy)
