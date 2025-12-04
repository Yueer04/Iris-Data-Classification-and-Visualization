# task2.py
# ===================================================
# 任务 2：两分类（三个特征）3D 决策边界可视化
# 标准化 + 放大点云
# 二分类：Setosa vs Non-Setosa
# ===================================================

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from classifier2d import set_chinese_font

def task2_3d_boundary():
    set_chinese_font()
    """绘制二分类 Logistic Regression 的 3D 决策边界"""
    
    # ---- 加载数据 ----
    iris = load_iris()
    X = iris.data[:, :3]                # 课件要求：三特征
    y = (iris.target != 0).astype(int)  # Setosa=0，其他=1（二分类）

    # ============================
    # 2. 标准化 + PCA 三维
    # ============================
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=3)
    X3 = pca.fit_transform(X_scaled)

    # --- 方案 A：放大点云，使点不再集中 ---
    scale_factor = 5.0
    X3 = X3 * scale_factor

    # ============================
    # 3. Logistic Regression 决策平面
    # ============================
    model = LogisticRegression().fit(X3, y)
    w = model.coef_[0]
    b = model.intercept_[0]

    x_range = np.linspace(X3[:, 0].min(), X3[:, 0].max(), 25)
    y_range = np.linspace(X3[:, 1].min(), X3[:, 1].max(), 25)
    xx, yy = np.meshgrid(x_range, y_range)

    # 平面:  w1*x + w2*y + w3*z + b = 0
    zz = -(w[0] * xx + w[1] * yy + b) / w[2]

    # ============================
    # 4. 绘图
    # ============================
    fig = plt.figure(figsize=(13, 11))
    ax = fig.add_subplot(111, projection='3d')

    # --- 决策平面 ---
    ax.plot_surface(
        xx, yy, zz,
        color="gray", alpha=0.35, edgecolor='none'
    )

    # --- 数据点 ---
    ax.scatter(X3[y == 0, 0], X3[y == 0, 1], X3[y == 0, 2],
               c='blue', s=40, alpha=0.95)
    ax.scatter(X3[y == 1, 0], X3[y == 1, 1], X3[y == 1, 2],
               c='red', s=40, alpha=0.95)

    # ============================
    # 坐标轴与标题
    # ============================
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("X3")

    ax.set_title("Task 2 — 3D Decision Boundary (Scaled PCA Space)")

    # PPT 风格视角
    ax.view_init(elev=20, azim=-60)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    task2_3d_boundary()
