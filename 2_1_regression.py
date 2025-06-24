import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    dfx, sy = load_diabetes(return_X_y=True, as_frame=True)

    print(f"---------- dfx :")
    print(dfx.head())
    print(f"---------- sy :")
    print(sy.head())
    print(f"---------- description :")
    print(dfx.describe())

    # Séparation du jeu d'entraînement et du jeu de test
    X_train, X_test, y_train, y_test = train_test_split(dfx, sy, test_size=0.3)

    # Initialiser et entraîner le modèle de régression linéaire
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Faire des prédictions sur l'ensemble de test
    y_pred = model.predict(X_test)
    print(f"---------- y_pred :")
    print(y_pred)

    # Utiliser la matrice de corrélation
    print(f"---------- matrice de corrélation :")
    print(dfx.corr())
    plt.figure(figsize=(20, 8))
    sns.heatmap(dfx.corr(), vmin=-1, vmax=1, annot=True, cmap='coolwarm', mask=np.triu(dfx.corr()))

    # **********************
    # Deuxième prédiction

    # Suppression de la colonne 's2'
    X_train = X_train.drop(columns=['s2'])
    X_test = X_test.drop(columns=['s2'])
    print(X_train.head())

    # Initialiser et entraîner le modèle de régression linéaire
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Faire des prédictions sur l'ensemble de test
    y_pred = model.predict(X_test)
    print(f"---------- y_pred :")
    print(y_pred)


    plt.show()