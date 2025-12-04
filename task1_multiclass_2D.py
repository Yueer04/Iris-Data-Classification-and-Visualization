# task1.py
# ===================================================
# 任务 1：多分类器三分类 2D 决策边界可视化
# 复用基础文件 classifier2d.py 中的：
#   - load_iris_2d()
#   - set_chinese_font()
# ===================================================

from classifier2d import load_iris_2d, plot_decision_boundary_2d, set_chinese_font
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from data_preview import load_iris_df

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# 原本任务数据加载
X, y = load_iris_2d()


import matplotlib.pyplot as plt

def task1_multiclass():
    set_chinese_font()

    """绘制多个分类器在2D（三分类）上的决策边界"""
    print("Preview of Iris DataFrame:")
    print(load_iris_df().head())
    # ---- 复用基础文件数据加载方式 ----
    X, y = load_iris_2d()

    # 数据集划分（与基础文件一致）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # ---- 定义多个分类器 ----
    models = {
        "Logistic Regression": LogisticRegression(max_iter=200),
        "SVM (RBF Kernel)": SVC(kernel='rbf', probability=True),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(max_depth=4)
    }

    # ---- 给每个模型分配不同概率颜色主题（蓝、绿、紫、青） ----
    row_cmaps = [plt.cm.Blues, plt.cm.BuGn, plt.cm.PuBu, plt.cm.GnBu]

    # ---- 决策边界颜色 (class0/class1/class2) ----
    decision_cmap = mcolors.ListedColormap(["#1f77b4", "#ff7f0e", "#2ca02c"])

    # ---- 生成网格 ----
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 300),
        np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 300)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    # ---- 大图区域 ----
    fig = plt.figure(figsize=(14, 15))

    # =======================
    # 左侧大概率色条
    # =======================
    cax_left = fig.add_axes([0.02, 0.25, 0.04, 0.55])
    dummy = np.linspace(0, 1, 256).reshape(256, 1)
    cax_left.imshow(dummy, cmap=plt.cm.Blues, aspect="auto")
    cax_left.set_xticks([])
    cax_left.set_yticks([0, 128, 255])
    cax_left.set_yticklabels(["0.0", "0.5", "1.0"])
    cax_left.set_title("Probability", fontsize=11)

    # =======================
    # 中间 4×4 图组
    # =======================
    grid_axes = []
    for row in range(4):
        for col in range(4):
            ax = fig.add_axes([
                0.10 + col * 0.21,
                0.78 - row * 0.135,
                0.20,
                0.13
            ])
            grid_axes.append(ax)

    # =======================
    # 填充 4×4 grid
    # =======================
    idx_plot = 0
    for (name, model), cmap in zip(models, row_cmaps):
        model.fit(X_train, y_train)
        proba = model.predict_proba(grid).reshape(300, 300, 3)

        # --- 前 3 列：class0 / class1 / class2 的概率图 ---
        for class_idx in range(3):
            ax = grid_axes[idx_plot]
            idx_plot += 1

            ax.imshow(
                proba[:, :, class_idx],
                extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                origin="lower",
                cmap=cmap,
                alpha=0.95
            )

            # 原始点散点
            ax.scatter(X[:, 0], X[:, 1], c=y, cmap=decision_cmap,
                       edgecolor="white", s=25, linewidth=0.3)

            # 左侧添加模型名
            if class_idx == 0:
                ax.set_ylabel(name, fontsize=9)

            # 顶部标题
            if idx_plot <= 4:
                ax.set_title(f"Class {class_idx}", fontsize=10)

            ax.set_xticks([])
            ax.set_yticks([])

        # --- 第 4 列：最大概率类别 ---
        ax = grid_axes[idx_plot]
        idx_plot += 1
        max_class = np.argmax(proba, axis=2)

        ax.imshow(
            max_class,
            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
            origin="lower",
            cmap=decision_cmap,
            alpha=0.75
        )

        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=decision_cmap,
                   edgecolor="white", s=25, linewidth=0.3)

        if idx_plot <= 4:
            ax.set_title("Max class", fontsize=10)

        ax.set_xticks([])
        ax.set_yticks([])

    # =======================
    # 底部三条概率 colorbars
    # =======================
    bottom_cmaps = ["Greens", "Oranges", "Blues"]
    bottom_titles = [
        "Probability class 2",
        "Probability class 1",
        "Probability class 0"
    ]

    for i in range(3):
        axb = fig.add_axes([0.25 + 0.17 * i, 0.05, 0.15, 0.10])
        dummy2 = np.linspace(0, 1, 256).reshape(1, 256)
        axb.imshow(dummy2, cmap=plt.get_cmap(bottom_cmaps[i]), aspect="auto")
        axb.set_xticks([0, 128, 255])
        axb.set_xticklabels(["0.0", "0.5", "1.0"])
        axb.set_yticks([])
        axb.set_title(bottom_titles[i], fontsize=9)    

    plt.show()


# 程序入口
if __name__ == "__main__":
    task1_multiclass()
