import copy
from matplotlib import pyplot as plt
import numpy as np
from sklearn import datasets, decomposition

def print_3d(X, y):
    
    np.random.seed(5)
    
    fig = plt.figure(1, figsize=(4, 3)) # 新增一張圖，編號為1，圖尺寸為4*3平方英寸
    plt.clf() # Clear figure => 清除圖表
    plt.cla() # Clear axis => 清除軸上的資料

    ax = fig.add_subplot(111, projection="3d", elev=48, azim=134)
    # add_subplot => 111 (row, col, index)
    ax.set_position([0, 0, 0.95, 1])

    for name, label in [("Setosa", 0), ("Versicolour", 1), ("Virginica", 2)]:
        ax.text3D(
            X[y == label, 0].mean(),
            X[y == label, 1].mean() + 1.5,
            X[y == label, 2].mean(),
            name,
            horizontalalignment="center",
            bbox=dict(alpha=0.5, edgecolor="w", facecolor="w"),
        )
    # Reorder the labels to have colors matching the cluster results
    y = np.choose(y, [1, 2, 0]).astype(float)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.nipy_spectral, edgecolor="k")

    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])

    plt.show()


if __name__ == "__main__":
    
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    
    pca = decomposition.PCA(n_components=3)
    pca.fit(X)
    X = pca.transform(X)
    print_3d(X, y)

