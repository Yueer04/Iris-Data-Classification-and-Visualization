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

def task3_3d_probability():
    set_chinese_font()
    """绘制二分类 Logistic Regression 的 3D 概率图"""
    
    # ============================
    # 1. 加载 Iris 数据（三个特征，二分类）
    # ============================
    iris = load_iris()
    X = iris.data[:, :3]               # 取前三个特征 x1, x2, x3
    y = (iris.target == 1).astype(int) # 只分类“Versicolor”(1) 其他为 0

    model = LogisticRegression().fit(X, y)

    # ============================
    # 2. 构建 x1-x2 网格，固定 x3
    # ============================
    x1 = np.linspace(X[:,0].min(), X[:,0].max(), 100)
    x2 = np.linspace(X[:,1].min(), X[:,1].max(), 100)
    X1g, X2g = np.meshgrid(x1, x2)

    # 固定第三个特征 x3 为其中位数
    x3_fixed = np.median(X[:,2])
    X3g = np.full_like(X1g, x3_fixed)

    # 组合成模型输入
    grid = np.c_[X1g.ravel(), X2g.ravel(), X3g.ravel()]

    # 预测分类为 1 的概率
    p1 = model.predict_proba(grid)[:, 1].reshape(X1g.shape)

    # ============================
    # ★★ 关键：将概率映射为 Z 轴高度（-100 ~ +100）★★
    # ============================
    Z = (p1 - 0.5) * 200

    # ============================
    # 3. 绘制与示例风格一致的 3D 图形
    # ============================
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # --- 中间蓝色 3D 曲面 ---
    ax.plot_surface(
        X1g, X2g, Z,
        cmap="Blues",
        edgecolor="none",
        alpha=0.6
    )

    # --- 蓝色 Wireframe（网格线）---
    ax.plot_wireframe(
        X1g, X2g, Z,
        color="navy",
        linewidth=0.5
    )

    # --- 左侧墙面概率等高线投影 ---
    ax.contourf(
        X1g, X2g, Z,
        zdir='x',
        offset=X[:,0].min() - 0.5,
        cmap="coolwarm",
        alpha=0.6
    )

    # --- 右侧墙面概率等高线投影 ---
    ax.contourf(
        X1g, X2g, Z,
        zdir='y',
        offset=X[:,1].max() + 0.5,
        cmap="coolwarm",
        alpha=0.6
    )

    # --- 底部平面概率等高线投影 ---
    ax.contourf(
        X1g, X2g, Z,
        zdir='z',
        offset=-120,
        cmap='coolwarm',
        alpha=0.85
    )

    # --- 顶部漂浮概率等高线投影 ---
    ax.contourf(
        X1g, X2g, Z,
        zdir='z',
        offset=120,
        cmap='coolwarm',
        alpha=0.55
    )

    # ============================
    # 4. 设置坐标轴范围（与示例保持一致）
    # ============================
    ax.set_xlim(X[:,0].min() - 0.5, X[:,0].max())
    ax.set_ylim(X[:,1].min(), X[:,1].max() + 0.5)
    ax.set_zlim(-120, 120)

    ax.set_xlabel("X1（Sepal Length）")
    ax.set_ylabel("X2（Sepal Width）")
    ax.set_zlabel("Z（概率映射高度）")

    ax.view_init(elev=30, azim=-60)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    task3_3d_probability()



