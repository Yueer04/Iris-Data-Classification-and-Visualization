# task1.py
# ===================================================
# 任务 1：多分类器三分类 2D 决策边界可视化
# 复用基础文件 classifier2d.py 中的：
#   - load_iris_2d()
#   - plot_decision_boundary_2d()
#   - set_chinese_font()
# ===================================================

from classifier2d import load_iris_2d, plot_decision_boundary_2d, set_chinese_font
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt

def task1_multiclass():
    set_chinese_font()

    """绘制多个分类器在2D（三分类）上的决策边界"""
    
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

    plt.figure(figsize=(14, 12))

    # ---- 四个子图 ----
    for idx, (name, model) in enumerate(models.items()):
        model.fit(X_train, y_train)

        ax = plt.subplot(2, 2, idx + 1)
        plot_decision_boundary_2d(model, X, y, title=name, ax=ax)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.suptitle("Task 1 — 多分类器三分类 2D 决策边界", fontsize=16)
    plt.show()


# ---- 程序入口 ----
if __name__ == "__main__":
    task1_multiclass()



