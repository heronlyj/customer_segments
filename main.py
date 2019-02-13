# # 检查你的Python版本
# from sys import version_info
# if version_info.major != 3:
#     raise Exception('请使用Python 3.x 来完成此项目')

# import matplotlib
# matplotlib.use('TkAgg')

# # 引入这个项目需要的库
# import numpy as np
# import pandas as pd
# import visuals as vs
# from IPython.display import display # 使得我们可以对DataFrame使用display()函数

# # 设置以内联的形式显示matplotlib绘制的图片（在notebook中显示更美观）
# # %matplotlib inline
# # 高分辨率显示
# # %config InlineBackend.figure_format='retina'

# # 载入整个客户数据集
# try:
#     data = pd.read_csv("customers.csv")
#     data.drop(['Region', 'Channel'], axis = 1, inplace = True)
#     print("Wholesale customers dataset has {} samples with {} features each.".format(*data.shape))
# except:
#     print("Dataset could not be loaded. Is the dataset missing?")


# display(data.describe())


# # TODO：从数据集中选择三个你希望抽样的数据点的索引
# indices = [75, 43, 312]

# # 为选择的样本建立一个DataFrame
# samples = pd.DataFrame(data.loc[indices], columns = data.keys()).reset_index(drop = True)
# print("Chosen samples of wholesale customers dataset:")
# display(samples)


# dropName = 'Milk'

# # TODO：为DataFrame创建一个副本，用'drop'函数丢弃一个特征# TODO： 
# new_data = data.drop(dropName, axis=1)

# # TODO：使用给定的特征作为目标，将数据分割成训练集和测试集
# import random
# random.seed(42)

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(new_data, data[dropName], test_size=0.25, random_state=42)

# # TODO：创建一个DecisionTreeRegressor（决策树回归器）并在训练集上训练它
# from sklearn.tree import DecisionTreeRegressor
# regressor = DecisionTreeRegressor(random_state=42)
# regressor.fit(X_train, y_train)

# # TODO：输出在测试集上的预测得分
# score = regressor.score(X_test, y_test)

# print(score)

# # TODO：使用自然对数缩放数据
# log_data = np.log(data)

# # TODO：使用自然对数缩放样本数据
# log_samples = np.log(samples)

# # 为每一对新产生的特征制作一个散射矩阵
# pd.plotting.scatter_matrix(log_data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');


# # 对于每一个特征，找到值异常高或者是异常低的数据点
# for feature in log_data.keys():
    
#     # TODO: 计算给定特征的Q1（数据的25th分位点）
#     Q1 = np.percentile(log_data[feature], q=25)
    
#     # TODO: 计算给定特征的Q3（数据的75th分位点）
#     Q3 = np.percentile(log_data[feature], q=75)
    
#     # TODO: 使用四分位范围计算异常阶（1.5倍的四分位距）
#     step = 1.5*(Q3 - Q1) #中度异常
    
#     # 显示异常点
#     print("Data points considered outliers for the feature '{}':".format(feature))
#     display(log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))])
    
# # TODO(可选): 选择你希望移除的数据点的索引
# outliers  = [66, 86, 95, 154, 161, 184, 338]

# # 以下代码会移除outliers中索引的数据点, 并储存在good_data中
# good_data = log_data.drop(log_data.index[outliers]).reset_index(drop = True)
# print(good_data)

# # TODO：通过在good data上进行PCA，将其转换成6个维度
# pca = None

# # TODO：使用上面的PCA拟合将变换施加在log_samples上
# pca_samples = None

# # 生成PCA的结果图
# pca_results = vs.pca_results(good_data, pca)


arr1 = [65, 6681, 95, 96, 128, 171, 193, 218, 304, 305, 338, 353, 355, 357, 412]
arr2 = [86, 98, 154, 356]
arr3 = [75, 154]
arr4 = [38, 57, 65, 145, 175, 264, 325, 420, 429, 439]
arr5 = [75, 161]
arr6 = [66, 109, 128, 137, 142, 154, 183, 184, 187, 203, 233, 285, 289, 343]

arr_result = []
drop_result = []

for a in [arr1, arr2, arr3, arr4, arr5, arr6]:
    for item in a:
        if item in arr_result and item not in drop_result:
            drop_result.append(item)
        elif item not in arr_result:
            arr_result.append(item)

arr_result.sort()
drop_result.sort()

print(arr_result)
print(drop_result)
