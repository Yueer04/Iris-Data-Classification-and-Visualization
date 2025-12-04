# task4.py
# ===================================================
# 任务 4：3D 决策边界 + 概率图（组合）
# 用颜色表示概率，同时叠加透明点云表示决策区域
# ===================================================

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from classifier2d import set_chinese_font

def task4_combined():
    set_chinese_font()
    """绘制 3D 决策边界 + 3D 概率图"""
    
    iris = load_iris()
    X = iris.data[:, :3]
    y = (iris.target != 0).astype(int)

    # ---- 模型 ----
    model = LogisticRegression(max_iter=200).fit(X, y)

    # ---- 网格 ----
    n = 22
    x = np.linspace(X[:, 0].min(), X[:, 0].max(), n)
    y_ = np.linspace(X[:, 1].min(), X[:, 1].max(), n)
    z = np.linspace(X[:, 2].min(), X[:, 2].max(), n)

    xx, yy, zz = np.meshgrid(x, y_, z)
    grid = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

    pred = model.predict(grid).reshape(xx.shape)
    prob = model.predict_proba(grid)[:, 1].reshape(xx.shape)

    # ---- 绘图 ----
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')

    # 概率图（颜色）
    sc = ax.scatter(xx, yy, zz, c=prob, cmap='viridis', alpha=0.4, s=20)
    fig.colorbar(sc, ax=ax, shrink=0.6, label="P(Class 1)")

    # 决策边界点
    ax.scatter(xx[pred == 0], yy[pred == 0], zz[pred == 0],
               color='blue', alpha=0.05, s=10)
    ax.scatter(xx[pred == 1], yy[pred == 1], zz[pred == 1],
               color='red', alpha=0.05, s=10)

    # 原始数据点
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y,
               cmap='bwr', edgecolor='k', s=80)

    ax.set_xlabel("Sepal Length")
    ax.set_ylabel("Sepal Width")
    ax.set_zlabel("Petal Length")
    ax.set_title("Task 4 — 3D Boundary + Probability Map")

    plt.show()


if __name__ == "__main__":
    task4_combined()


