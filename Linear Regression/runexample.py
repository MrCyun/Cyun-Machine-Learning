import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import LinearRegression
# 读取数据
data = pd.read_csv('linear regression/california housing price.csv')
# 处理缺失值
# 填充'total_bedrooms'的缺失值
data['total_bedrooms'] = data['total_bedrooms'].fillna(data['total_bedrooms'].median())
# 特征选择与目标变量提取
X = data.drop('median_house_value', axis=1)  # 选择所有特征列
y = data['median_house_value']  # 目标变量是房价
# 创建预处理流水线
# 对于数值型特征进行标准化
# 对于文本特征 'ocean_proximity' 使用独热编码
numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
categorical_features = ['ocean_proximity']
# 数值特征标准化
numeric_transformer = StandardScaler()
# 类别特征独热编码
categorical_transformer = OneHotEncoder(handle_unknown='ignore')
# 将所有的转换步骤组合成一个预处理管道
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
# 创建最终的Pipeline，它包含预处理和模型（线性回归）
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
])
# 对数据进行预处理
X_processed = pipeline.fit_transform(X)
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
# 查看训练集和测试集的形状
print(f'训练集特征形状: {X_train.shape}')
print(f'测试集特征形状: {X_test.shape}')
print(f'训练集目标变量形状: {y_train.shape}')
print(f'测试集目标变量形状: {y_test.shape}')

model = LinearRegression.linear_regression()
model.fit(X_train, y_train)
# 预测
y_pred = model.predict(X_test)
# 评估模型
y_pred = y_pred.flatten()
y_test = y_test.to_numpy()
# MAE
mae = np.mean(np.abs(y_test - y_pred))
print("MAE: ", mae)
# R-squared
ss_residuals = np.sum((y_test - y_pred)**2)
ss_tot = np.sum((y_test - np.mean(y_test))**2)
r2 = 1 - ss_residuals / ss_tot
print("R-squared: ", r2)

# 对比SKlearn自带的线性回归模型
print("对比SKlearn自带的线性回归模型")
from sklearn.linear_model import LinearRegression
sklearn_model = LinearRegression()
sklearn_model.fit(X_train, y_train)
y_pred = sklearn_model.predict(X_test)
y_pred = y_pred.flatten()
# MAE
mae = np.mean(np.abs(y_test - y_pred))
print("MAE: ", mae)
# R-squared
ss_residuals = np.sum((y_test - y_pred)**2)
ss_tot = np.sum((y_test - np.mean(y_test))**2)
r2 = 1 - ss_residuals / ss_tot
print("R-squared: ", r2)






