import numpy as np
import pandas as pd
from seaborn.matrix import dendrogram
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine, load_iris
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

if __name__ == '__main__':
    # Charger le jeu de données iris
    iris = load_iris()
    X = iris.data
    feature_names = iris.feature_names

    K_range = range(2, 11)
    silhouette_scores = []

    for i_cluster in K_range:
        kmeans = KMeans(n_clusters=i_cluster, random_state=42)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        silhouette_scores.append(score)

    print(f"Silhouette scores : {silhouette_scores}")

    # Trouver la valeur optimale de K qui maximise le score de silhouettte
    optimal_k = K_range[np.argmax(silhouette_scores)]
    best_score = max(silhouette_scores)

    # Afficher le nombre de clusters optimal et le score correspondant
    print(f"Optimal k : {optimal_k}")
    print(f"Best score : {best_score:.2f}")
