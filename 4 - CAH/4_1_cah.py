import pandas as pd
from seaborn.matrix import dendrogram
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

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

    fig = plt.figure(figsize=(15, 25))

    dendrogram = sch.dendrogram(sch.linkage(X_scaled, method="ward"))

    plt.title('Dendrogramme pour le jeu de données Wine')
    plt.xlabel('Index des échantillons')
    plt.ylabel('Distance')
    plt.show()