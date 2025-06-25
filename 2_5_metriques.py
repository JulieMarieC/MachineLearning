import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

if __name__ == '__main__':
    dfX, sy = load_diabetes(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(dfX, sy, test_size=0.3)

    # REGRESSION LINEAIRE
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculer les métriques de performances
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)

    # Afficher les résultats
    print(f"--- REGRESSION LINEAIRE")
    print(f"MSE: {mse:.2f}")
    print(f"Coefficient de détermination (R²): {r2:.2f}")
    print(f"RMSE: {rmse:.2f}")

    # RANDOMFOREST
    rf_regressor = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    rf_regressor.fit(X_train, y_train)
    y_pred = rf_regressor.predict(X_test)

    # Calculer les métriques de performances
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)

    # Afficher les résultats
    print(f"--- RANDOM FOREST")
    print(f"MSE: {mse:.2f}")
    print(f"Coefficient de détermination (R²): {r2:.2f}")
    print(f"RMSE: {rmse:.2f}")

