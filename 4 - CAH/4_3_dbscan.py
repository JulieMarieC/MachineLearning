import pandas as pd
from seaborn.matrix import dendrogram
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

if __name__ == '__main__':
    # Charger le jeu de données wine
    wine = load_wine()
    X = wine.data
    feature_names = wine.feature_names

    print(f"X : {X}")
    print(f"feature_names : {feature_names}")

    # Normaliser les données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled)
    print(f"X_scaled : {X_scaled}")

    clusters = DBSCAN(eps=2, min_samples=8).fit(X_scaled)
    print(f"clusters.labels_ {clusters.labels_}")
    set(clusters.labels_)
    y_dbscan = DBSCAN(eps=2, min_samples=8).fit_predict(X_scaled)

    clusters1 = pd.DataFrame(y_dbscan)
    clusters1.columns = ['clusters']

    X_scaled.columns = feature_names
    data_final = pd.concat([X_scaled, clusters1], axis=1)
    print(f"data_final : {data_final}")

