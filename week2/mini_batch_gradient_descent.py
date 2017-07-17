# coding=utf-8
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import normalize
import random

# 构造训练数据
original_data = pd.read_csv(
        "pima-indians-diabetes.data.csv",
        sep=',',
        engine='python')

x = original_data.values
m = min(len(x[0]), random.randint(50, 256))  # 训练数据点数目
input_data = x[:, :-1]
features = len(input_data[0])
target_data = x[:, -1:]

# 两种终止条件
loop_max = 10000  # 最大迭代次数(防止死循环)
epsilon = 1e-3

# 初始化权值
np.random.seed(0)
w = np.random.randn(features)

alpha = 0.001  # 步长
diff = 0.
error = np.zeros(features)
count = 0  # 循环次数
finish = 0  # 终止标志
error_list = []

# -------------------------------------------mini-BGD----------------------------------------------------------
while count < loop_max:
    count += 1

    # 遍历训练数据集，不断更新权值
    sum_m = np.zeros(features)
    for i in range(m):
        diff = (np.dot(w, input_data[i]) - target_data[i]) * target_data[i]  # 训练集代入,计算误差值
        # 采用min-BGD,对部分样例求和
        sum_m = sum_m + diff

    w = w - alpha * sum_m
    error_list.append(np.sum(sum_m)**2)
    if np.linalg.norm(w - error) < epsilon:     # 终止条件：前后两次计算出的权向量的绝对误差充分小
        finish = 1
        break
    else:
        error = w
print 'loop count = %d' % count,  '\tw:[%f, %f]' % (w[0], w[1])

# ----------------------------------------------------------------------------------------------------------------------


# check with scipy linear regression
slope, intercept, r_value, p_value, slope_std_error = stats.linregress(x, target_data)
print 'intercept = %s slope = %s' % (intercept, slope)

plt.plot(range(len(error_list[0:100])), error_list[0:100])
plt.show()

plt.plot(x, target_data, 'k+')
plt.plot(x, w[1] * x + w[0], 'r')
plt.show()