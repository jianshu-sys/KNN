#!/usr/bin/python
# coding:utf8

"""
Created on 2017-06-28
Updated on 2017-06-28
KNN: k近邻算法
Author: 小瑶
GitHub: https://github.com/apachecn/AiLearning
"""
#即使是在Python2.7版本的环境下，print功能的使用格式也遵循Python3.x版本中的加括号的形式。
from __future__ import print_function
#输出文件开头注释的内容
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from numpy import *
from matplotlib.colors import ListedColormap
from sklearn import neighbors

n_neighbors = 3

# 导入一些要玩的数据

X = array([[-1.0, -1.1], [-1.0, -1.0], [0, 0], [1.0, 1.1], [2.0, 2.0], [2.0, 2.1]])
y = array([0, 0, 0, 1, 1, 1])

h = .02  # 网格中的步长

# 创建彩色的地图
#这个表示颜色：cmap_light：表示背景颜色，cmap_bold：数据点的颜色
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00'])

# 我们创建了一个knn分类器的实例，并适合数据。
clf = neighbors.KNeighborsClassifier(n_neighbors, weights="distance")
clf.fit(X, y)

# 绘制决策边界。为此，我们将为每个分配一个颜色
# 来绘制网格中的点 [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # 将结果放入一个彩色图中
Z = Z.reshape(xx.shape)
plt.figure()
#用来绘制分类图，也就是背景分类
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
plt.show()
# 绘制训练点
#x，y：表示的是数组---我们即将绘制散点图的数据点
#c:表示的是颜色
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
#x轴y轴的上下限
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
#标题
plt.title("3-Class classification (k = %i, weights = distance)"
              % (n_neighbors))

plt.show()