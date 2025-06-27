import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

if __name__ == "__main__":
    dfX, sy = load_diabetes(return_X_y=True, as_frame=True)

    X_train, X_test, y_train, y_test = train_test_split(dfX, sy, test_size=0.3)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),           # Etape de standardisation des données
        ('regressor', LinearRegression())       # Etape de régression linéaire
    ])

    # Entraîner la pipeline sur les données d'entraînement
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    print(f"Coefficient de détermination (R²) : {r2:0.2f}")

