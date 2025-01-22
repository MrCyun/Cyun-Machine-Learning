import pandas as pd
import numpy as np
import NaiveBayesian
from collections import Counter
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score   # AUC指标
from sklearn.metrics import confusion_matrix    # 混淆矩阵

#读取adult.data、adult.test文件
df_train_set = pd.read_csv('Naive Bayesian/adult.data')
df_test_set = pd.read_csv('Naive Bayesian/adult.test')
#新加一行作为索引
df_train_set.columns = ['age', 'workclass','fnlwgt',
              'education','educationnum','maritalstatus',
              'occupation','relationship','race',
              'sex','capitalgain','capital-loss',
              'hoursperweek','nativecountry','income']
df_test_set.columns = ['age', 'workclass','fnlwgt',
              'education','educationnum','maritalstatus',
              'occupation','relationship','race',
              'sex','capitalgain','capital-loss',
              'hoursperweek','nativecountry','income']
# 删除无关属性
df_train_set.drop(['fnlwgt',"education", "nativecountry"], axis=1, inplace=True)
df_test_set.drop(['fnlwgt',"education", "nativecountry"], axis=1, inplace=True)
#进行数据清洗
for i in df_train_set.columns:
    df_train_set[i].replace('?', 'Unknown', inplace=True)
    df_test_set[i].replace('?', 'Unknown', inplace=True)
    for col in df_train_set.columns:
        if df_train_set[col].dtype != 'int64':
           df_train_set[col] = df_train_set[col].apply(lambda val: val.replace(" ", ""))
           df_train_set[col] = df_train_set[col].apply(lambda val: val.replace(".", ""))
           df_test_set[col] = df_test_set[col].apply(lambda val: val.replace(" ", ""))
           df_test_set[col] = df_test_set[col].apply(lambda val: val.replace(".", ""))
#将连续型变量（age、educationum）转化为离散的区间
colnames = list(df_train_set.columns)	
colnames.remove('age')
colnames.remove('educationnum')
colnames = ['ageGroup', 'eduGroup']+colnames
labels = ["{0}-{1}".format(i, i+9) for i in range(0,100,10)]
df_train_set['ageGroup'] = pd.cut(df_train_set.age, range(0,101,10), right = False, labels = 	labels) 
df_test_set['ageGroup'] = pd.cut(df_test_set.age, range(0,101,10), right = False, labels = 	labels) 
labels = ["{0}-{1}".format(i,i+4) for i in range(0,20,5)]    
df_train_set['eduGroup'] = pd.cut(df_train_set.educationnum, range(0,21,5), right = False, labels = 	labels)   
df_test_set['eduGroup'] = pd.cut(df_test_set.educationnum, range(0,21,5), right = False, labels = 	labels)
df_train_set = df_train_set[colnames] 
df_test_set = df_test_set[colnames]
#将非数值型数据转换为数值型数据借用sklearn_pandas包中的DataFrameMapper类
mapper = DataFrameMapper([('ageGroup', LabelEncoder()),('eduGroup', LabelEncoder()),
                          ('workclass', LabelEncoder()),('maritalstatus',LabelEncoder()),
                          ('occupation', LabelEncoder()),('relationship',LabelEncoder()),
                          ('race', LabelEncoder()),('sex', LabelEncoder()),
                          ('income', LabelEncoder())], df_out=True, default=None)
cols = list(df_train_set.columns)
cols.remove('income')
cols = cols[:-3]+['income']+cols[-3:]  
#调用fit_transform()方法拟合数据，并标准化
#替换表头，移除样本标记income
df_train = mapper.fit_transform(df_train_set.copy())
df_train.columns = cols
df_test = mapper.transform(df_test_set.copy())
df_test.columns = cols
cols.remove('income')
X_train, y_train = df_train[cols].values, df_train['income'].values
X_test, y_test = df_test[cols].values, df_test['income'].values

# 计算训练集和测试集的样本数量
print(f"训练集特征形状: {X_train.shape}")
print(f"测试集特征形状: {X_test.shape}")
print(f"训练集目标变量形状: {y_train.shape}")
print(f"测试集目标变量形状: {y_test.shape}")

# 训练 
model = NaiveBayesian.NaiveBayesian()
model.fit(X_train, y_train)
# 预测
y_pred = model.predict(X_test)
# 评估
# 计算准确率
accuracy = np.sum(y_pred == y_test)/len(y_test)*100
print(f"Accuracy: {accuracy:.2f}%")
# 计算AUC指标
auc = roc_auc_score(y_test, y_pred)
print(f"AUC: {auc:.4f}")
# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{cm}")

# 对比SKlearn中的朴素贝叶斯分类器
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = np.sum(y_pred == y_test)/len(y_test)*100
print(f"Accuracy: {accuracy:.2f}%")
auc = roc_auc_score(y_test, y_pred)
print(f"AUC: {auc:.4f}")
cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{cm}")









