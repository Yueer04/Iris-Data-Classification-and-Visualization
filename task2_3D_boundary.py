# task2.py
# ===================================================
# 任务 2：两分类（三个特征）3D 决策边界可视化
# 二分类：Setosa vs Non-Setosa
# ===================================================

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from classifier2d import set_chinese_font
set_chinese_font()


def task2_3d_boundary():
    """绘制二分类 Logistic Regression 的 3D 决策边界"""
    
    # ---- 加载数据 ----
    iris = load_iris()
    X = iris.data[:, :3]       # 使用前三个特征
    y = (iris.target != 0).astype(int)  # Setosa=0，其他=1（二分类）

    # ---- 训练模型 ----
    model = LogisticRegression(max_iter=200).fit(X, y)

    # ---- 创建 3D 网格 ----
    n = 25  # 网格点数量
    x = np.linspace(X[:, 0].min(), X[:, 0].max(), n)
    y_ = np.linspace(X[:, 1].min(), X[:, 1].max(), n)
    z = np.linspace(X[:, 2].min(), X[:, 2].max(), n)

    xx, yy, zz = np.meshgrid(x, y_, z)

    grid = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    pred = model.predict(grid)  # 每个网格点的预测分类
    pred = pred.reshape(xx.shape)

    # ---- 绘图 ----
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 决策边界点（使用透明小点模拟 3D 决策面）
    ax.scatter(xx[pred == 0], yy[pred == 0], zz[pred == 0], s=8,
               color='blue', alpha=0.05)
    ax.scatter(xx[pred == 1], yy[pred == 1], zz[pred == 1], s=8,
               color='red', alpha=0.05)

    # 原始数据点
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, s=70,
               cmap='bwr', edgecolor='k')

    ax.set_xlabel("Sepal Length")
    ax.set_ylabel("Sepal Width")
    ax.set_zlabel("Petal Length")
    ax.set_title("Task 2 — 3D Decision Boundary (Binary Classification)")

    plt.show()


if __name__ == "__main__":
    task2_3d_boundary()

