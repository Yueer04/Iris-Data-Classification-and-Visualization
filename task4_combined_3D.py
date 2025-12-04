# task4.py
# ===================================================
# 任务 4：三分类（三特征）
# 3D 决策边界 + 三分类概率图（四图合一）
# ===================================================

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from classifier2d import set_chinese_font


def task4_combined():
    set_chinese_font()

    # ============================
    # 1. 加载 iris（三分类）
    # ============================
    iris = load_iris()
    X = iris.data[:, :3]     # 三特征
    y = iris.target          # 0/1/2

    # ============================
    # 2. 三分类 Logistic Regression（softmax）
    # ============================
    model = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=500
    ).fit(X, y)

    # ============================
    # 3. 构建 3D 网格
    # ============================
    n = 22
    x = np.linspace(X[:, 0].min(), X[:, 0].max(), n)
    y_ = np.linspace(X[:, 1].min(), X[:, 1].max(), n)
    z = np.linspace(X[:, 2].min(), X[:, 2].max(), n)

    xx, yy, zz = np.meshgrid(x, y_, z)
    grid = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

    # 三分类 softmax 概率
    probs = model.predict_proba(grid)
    P0 = probs[:, 0].reshape(xx.shape)
    P1 = probs[:, 1].reshape(xx.shape)
    P2 = probs[:, 2].reshape(xx.shape)

    # ============================
    # 4. 三条决策边界：P(i) = P(j)
    # ============================
    eps = 0.02
    boundary_01 = np.abs(P0 - P1) < eps
    boundary_02 = np.abs(P0 - P2) < eps
    boundary_12 = np.abs(P1 - P2) < eps

    # ============================
    # 5. 四图合一（2×2）
    # ============================
    fig = plt.figure(figsize=(18, 16))

    # ----------------------------------------
    # 图 1：三分类三条决策边界
    # ----------------------------------------
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')

    ax1.scatter(xx[boundary_01], yy[boundary_01], zz[boundary_01],
                s=5, color="purple", alpha=0.4, label="Boundary 0 vs 1")
    ax1.scatter(xx[boundary_02], yy[boundary_02], zz[boundary_02],
                s=5, color="green", alpha=0.4, label="Boundary 0 vs 2")
    ax1.scatter(xx[boundary_12], yy[boundary_12], zz[boundary_12],
                s=5, color="orange", alpha=0.4, label="Boundary 1 vs 2")

    ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=y,
                cmap="viridis", edgecolor='k', s=70)

    ax1.set_title("Decision Boundaries")
    ax1.set_xlabel("X1")
    ax1.set_ylabel("X2")
    ax1.set_zlabel("X3")
    ax1.legend()

    # ----------------------------------------
    # 图 2：P(class 0)
    # ----------------------------------------
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    sc2 = ax2.scatter(xx, yy, zz, c=P0, cmap='coolwarm',
                      s=15, alpha=0.5)
    fig.colorbar(sc2, ax=ax2, shrink=0.6)
    ax2.scatter(X[:, 0], X[:, 1], X[:, 2],
                c=y, cmap="viridis", edgecolor='k', s=50)
    ax2.set_title("P(Class 0)")

    # ----------------------------------------
    # 图 3：P(class 1)
    # ----------------------------------------
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    sc3 = ax3.scatter(xx, yy, zz, c=P1, cmap='coolwarm',
                      s=15, alpha=0.5)
    fig.colorbar(sc3, ax=ax3, shrink=0.6)
    ax3.scatter(X[:, 0], X[:, 1], X[:, 2],
                c=y, cmap="viridis", edgecolor='k', s=50)
    ax3.set_title("P(Class 1)")

    # ----------------------------------------
    # 图 4：P(class 2)
    # ----------------------------------------
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    sc4 = ax4.scatter(xx, yy, zz, c=P2, cmap='coolwarm',
                      s=15, alpha=0.5)
    fig.colorbar(sc4, ax=ax4, shrink=0.6)
    ax4.scatter(X[:, 0], X[:, 1], X[:, 2],
                c=y, cmap="viridis", edgecolor='k', s=50)
    ax4.set_title("P(Class 2)")

    plt.suptitle("Task 4 — 三分类三特征：决策边界 + 3D 概率图", fontsize=18)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    task4_combined()
