import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import scipy as sp
import warnings
warnings.filterwarnings("ignore")

# 工具函数
''''' 均方误差根 '''
def rmse(y_test, y):
    return sp.sqrt(sp.mean((y_test - y) ** 2))

''''' 与均值相比的优秀程度，介于[0~1]。0表示不如均值。1表示完美预测.这个版本的实现是参考scikit-learn官网文档  '''
'''
    R2 score的计算公式函数
'''
def R2(y_test, y_true):
    return 1 - ((y_test - y_true) ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum()

# 主函数的入口
if __name__ == '__main__':
    # 读取数据
    data_true = pd.read_csv('data.csv')
    # 分别截取训练数据的特征x和两个预测目标y1和y2
    data_x = data_true.drop(['A(%)','B(%)'],axis=1)
    data_y1 = data_true['A(%)']
    data_y2 = data_true['B(%)']
    # 处理第一列的时间数据，将年、月、日、时分别作为特征项
    data_x = pd.concat([data_x, data_x['time'].str.split('-', expand = True)], axis = 1, names = ['year','month','day','hour'])
    data_x = data_x.drop(['time'],axis = 1)
    # 重命名新增列
    data_x = data_x.rename(columns={0:'year',1:'month',2:'day',3:'hour'})
    """
        1.将数据集划分训练集和测试集---针对A(%)那一列
    """
    train_X, test_X, train_y, test_y = train_test_split(data_x,data_y1,test_size=0.2,random_state=0)
    # 模型训练
    regr = DecisionTreeRegressor(max_depth=5)
    regr.fit(train_X,train_y)
    # 得到模型预测结果
    y_test = regr.predict(test_X)
    # 输出评价结果
    print("针对A(%)这一列的回归预测的效果评估指标如下：")
    print("rmse:  " + str(rmse(y_test, test_y)))
    print("R2:  " + str(R2(y_test, test_y)))
    # 将测试集的预测结果写入excel文件
    data_excel = pd.concat([test_X.reset_index(),test_y.reset_index()['A(%)'],pd.DataFrame(y_test)],axis=1)
    data_excel = data_excel.rename(columns={0: '预测的A(%)'})
    data_excel = data_excel.drop(['index'],axis = 1)
    data_excel.to_excel('A(%)的测试集预测结果文件.xlsx')

    """
           2.将数据集划分训练集和测试集---针对B(%)那一列
       """
    train_X, test_X, train_y, test_y = train_test_split(data_x, data_y2, test_size=0.2, random_state=0)
    # 模型训练
    regr = DecisionTreeRegressor(max_depth=5)
    regr.fit(train_X, train_y)
    # 得到模型预测结果
    y_test = regr.predict(test_X)
    # 输出评价结果
    print()
    print("针对B(%)这一列的回归预测的效果评估指标如下：")
    print("rmse:  " + str(rmse(y_test, test_y)))
    print("R2:  " + str(R2(y_test, test_y)))
    # 将测试集的预测结果写入excel文件
    data_excel = pd.concat([test_X.reset_index(), test_y.reset_index()['B(%)'], pd.DataFrame(y_test)], axis=1)
    data_excel = data_excel.rename(columns={0: '预测的B(%)'})
    data_excel = data_excel.drop(['index'], axis=1)
    data_excel.to_excel('B(%)的测试集预测结果文件.xlsx')

    print(data_excel)





