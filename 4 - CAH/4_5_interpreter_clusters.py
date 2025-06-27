import pandas as pd
from seaborn.matrix import dendrogram
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

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

    kmeans = KMeans(n_clusters=3)
    y_kmeans = kmeans.fit_predict(X_scaled)

    clusters = pd.DataFrame(y_kmeans)
    clusters.columns = ['clusters']
    clusters_labels = kmeans.labels_

    X_scaled.columns = feature_names
    data_final = pd.concat([X_scaled, clusters], axis=1)
    print(f"data_final : {data_final}")

    # Score de silhouette
    silhouette_avg = silhouette_score(X_scaled, clusters_labels)
    print(f"silhouette_score : {silhouette_avg:.2f}")

    fig = plt.figure(figsize=(13, 3))
    sns.heatmap(pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_[:,:]),
                             columns=X_scaled.columns),annot=True)
    plt.show()

    sns.boxplot(x=data_final['clusters'], y=data_final['magnesium'], palette='Set2')
    plt.title("Distribution du magnésium par cluster")
    plt.show()