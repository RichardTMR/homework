# coding=utf-8

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import random
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 构造训练数据
original_data = pd.read_csv(
    "pima-indians-diabetes.data.csv",
    sep=',',
    engine='python').values

train_data, test_data, target_data, test_target = train_test_split(
    original_data[:, :-1], original_data[:, -1:], train_size=0.75)

whole_input = np.c_[np.ones(len(train_data)), train_data]
# input_data = train_data
features = len(whole_input[0])

# 两种终止条件
loop_max = 10000  # 最大迭代次数(防止死循环)
epsilon = 1e-3

# 初始化权值
np.random.seed(0)
w = np.random.randn(features)

alpha = 0.0005  # 步长
diff = 0.
error = np.zeros(features)
count = 0  # 循环次数
finish = 0  # 终止标志
error_list = []
input_data = whole_input
random.shuffle(input_data)
m = input_data.size
batch_size = min(m, 50)

# -------------------------------------------mini-BGD----------------------------------------------------------
while count < loop_max:
    count += 1

    # 遍历训练数据集，不断更新权值
    sum_m = np.zeros(features)
    for i in range(0, target_data.size):
        diff = (np.dot(input_data[i], w) - target_data[i]) * target_data[i]  # 训练集代入,计算误差值
        # 采用min-BGD,对部分样例求和
        sum_m = sum_m + diff

        if i % batch_size == 0:
            w = w - alpha * (sum_m / batch_size)
            error_list.append(1.0 / (2 * m) * np.sum(sum_m) ** 2)
            sum_m = np.zeros(features)
            if np.linalg.norm(w - error) < epsilon:  # 终止条件：前后两次计算出的权向量的绝对误差充分小
                finish = 1
                break
            else:
                error = w

# ----------------------------------------------------------------------------------------------------------------------

regressor = LinearRegression()
regressor.fit(train_data, target_data)

print regressor.score(test_data, test_target)

plt.plot(range(len(error_list[0:100])), error_list[0:100])
plt.show()
