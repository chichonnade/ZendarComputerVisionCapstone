from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from mpl_toolkits.mplot3d import Axes3D


def run_cluster(X_radar):

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X_radar[:,0], X_radar[:,1], X_radar[:,2])
    plt.show()

    X = X_radar
    db = DBSCAN(eps=0.3, min_samples=10).fit(X)
    core_samples = db.core_sample_indices_
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    clusters = [X[labels == i] for i in range(n_clusters_)]
    outliers = X[labels == -1]
    unique_labels = set(labels)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = labels == k
        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], "o", markerfacecolor=tuple(col), markeredgecolor="k", markersize=14,)
        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], "o", markerfacecolor=tuple(col), markeredgecolor="k", markersize=6,)

    plt.title(f"Estimated number of clusters: {n_clusters_}")
    plt.show()


def run_cluster_2(data):

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=300)
    ax.view_init(azim=200)
    plt.show()

    model = DBSCAN(eps=2.5, min_samples=2)
    model.fit_predict(data)
    pred = model.fit_predict(data)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=model.labels_, s=300)
    ax.view_init(azim=200)
    plt.show()


def run_cluster_3(data):
    # Assuming all_camera is a dictionary with keys 'ZRVE1001', 'ZRVE1002', 'ZRVC2001'
    # Grid search for best hyperparameters
    eps_values = np.arange(0.1, 5, 0.2)  # Adjust the range and step as needed
    min_samples_values = range(2, 6)  # Adjust the range and step as needed
    best_score = -1
    best_eps = None
    best_min_samples = None
    X = data

    for eps in eps_values:
        for min_samples in min_samples_values:
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
            labels = db.labels_
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters_ == 0:
                continue
            #compute silhouette score for the current hyperparameters
            silhouette_score = metrics.silhouette_score(X, labels)
            if silhouette_score > best_score:
                best_score = silhouette_score
                best_eps = eps
                best_min_samples = min_samples

    # Using the best parameters
    db = DBSCAN(eps=best_eps, min_samples=best_min_samples).fit(X)
    labels = db.labels_
    print(f"Silhouette Coefficient: {best_score:.3f}")
    print(f"Best eps: {best_eps}")
    print(f"Best min_samples: {best_min_samples}")
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)

    ## Plotting
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')

    ## Filter Points out that are not in image
    points = data
    filtered_indices = np.where((0 < points[:, 0]) & (points[:, 0] < 1616) & (0 < points[:, 1]) & (points[:, 1] < 1240))
    filtered_points = points[filtered_indices]
    filtered_labels = np.array(labels)[filtered_indices]
    unique_labels = set(np.array(labels))
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    cluster_dict = []
    for label, col in zip(unique_labels, colors):
        if label == -1:
            continue
        cluster_points = points[np.array(labels) == label, :]
        x, y, z = cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2]
        cluster_dict.append([x, y, z])
        ax.scatter(x, y, z, c=[col], label=f'Cluster {label}', alpha=0.8)

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    plt.title('3D Points by Clusters (excluding noise)')

    ax.view_init(elev=35, azim=90)
    ax.set_xlim(-20, 20)
    ax.set_ylim(-10, 10)
    ax.set_zlim(20, 80)

    return points,unique_labels,labels,cluster_dict
