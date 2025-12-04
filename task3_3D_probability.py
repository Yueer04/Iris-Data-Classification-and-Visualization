# task3.py
# ===================================================
# 任务 3：两分类（三个特征）3D 概率图可视化
# 显示每个点属于“类 1” 的概率（使用颜色表达）
# ===================================================

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from classifier2d import set_chinese_font
set_chinese_font()

def task3_3d_probability():
    """绘制二分类 Logistic Regression 的 3D 概率图"""
    
    # ---- 数据 ----
    iris = load_iris()
    X = iris.data[:, :3]
    y = (iris.target != 0).astype(int)

    # ---- 模型 ----
    model = LogisticRegression(max_iter=200).fit(X, y)

    # ---- 生成 3D 网格 ----
    n = 25
    x = np.linspace(X[:, 0].min(), X[:, 0].max(), n)
    y_ = np.linspace(X[:, 1].min(), X[:, 1].max(), n)
    z = np.linspace(X[:, 2].min(), X[:, 2].max(), n)

    xx, yy, zz = np.meshgrid(x, y_, z)
    grid = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

    # ---- 预测概率 ----
    prob = model.predict_proba(grid)[:, 1].reshape(xx.shape)

    # ---- 绘制概率点云 ----
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(xx, yy, zz, c=prob, cmap='coolwarm',
                    alpha=0.6, s=18)
    fig.colorbar(sc, ax=ax, shrink=0.6, label="P(Class 1)")

    # 原始数据点
    ax.scatter(X[:, 0], X[:, 1], X[:, 2],
               c=y, cmap='bwr', edgecolor='k', s=70)

    ax.set_xlabel("Sepal Length")
    ax.set_ylabel("Sepal Width")
    ax.set_zlabel("Petal Length")
    ax.set_title("Task 3 — 3D Probability Map")

    plt.show()


if __name__ == "__main__":
    task3_3d_probability()

