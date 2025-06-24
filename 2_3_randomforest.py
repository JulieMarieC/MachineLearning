import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns

if __name__ == '__main__':
    dfX, sy = load_diabetes(return_X_y=True, as_frame=True)

    # Séparation jeu d'entraînement et jeu de test
    X_train, X_test, y_train, y_test = train_test_split(dfX, sy, test_size=0.3)

    # Initialiser et entraîner le modèle de régression avec une forêt aléatoire
    rf_regressor = RandomForestRegressor(random_state=42)
    rf_regressor.fit(X_train, y_train)

    # Faire des prédictions sur l'ensemble de test
    y_pred = rf_regressor.predict(X_test)

    print(f"Prédiction 1 : {y_pred}")

    # Initialiser et entraîner le modèle de régression avec une forêt aléatoire
    rf_regressor = RandomForestRegressor(n_estimators=80, max_depth=5, random_state=42)
    rf_regressor.fit(X_train, y_train)

    # Faire des prédictions sur l'ensemble de test
    y_pred = rf_regressor.predict(X_test)

    print(f"Prédiction 2 : {y_pred}")
