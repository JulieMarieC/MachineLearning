import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import seaborn as sns

if __name__ == '__main__':
    dfX, sy = load_diabetes(return_X_y=True, as_frame=True)

    # Séparation jeu d'entraînement et jeu de test
    X_train, X_test, y_train, y_test = train_test_split(dfX, sy, test_size=0.3)

    # Initialiser et entraîner le modèle de régression avec un arbre de décision
    tree_model = DecisionTreeRegressor(random_state=42)
    tree_model.fit(X_train, y_train)

    # Faire des prédictions sur l'ensemble de test
    y_pred = tree_model.predict(X_test)
    print(f"Prédiction 1: {y_pred}")

    # Initialiser et entraîner le modèle de régression avec un arbre de décision
    tree_model = DecisionTreeRegressor(max_depth=5, min_samples_split=10, random_state=42)
    tree_model.fit(X_train, y_train)

    # Faire des prédictions sur l'ensemble de test
    y_pred = tree_model.predict(X_test)
    print(f"Prédiction 2: {y_pred}")

