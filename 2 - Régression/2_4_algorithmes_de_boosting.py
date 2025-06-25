import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
import seaborn as sns
from xgboost import XGBRegressor

if __name__ == '__main__':
    dfX, sy = load_diabetes(return_X_y=True, as_frame=True)

    # Séparation jeu d'entraînement et jeu de test
    X_train, X_test, y_train, y_test = train_test_split(dfX, sy, test_size=0.3)

    # 1 AdaBoostRegressor
    adaboost_regressor = AdaBoostRegressor(n_estimators=50, learning_rate=1)
    adaboost_regressor.fit(X_train, y_train)
    y_pred = adaboost_regressor.predict(X_test)
    print(f"AdaBoostRegressor : {y_pred}")

    # 2 XGBoost
    xgboost_regressor = XGBRegressor(n_estimators=50, learning_rate=1)
    xgboost_regressor.fit(X_train, y_train)
    y_pred = xgboost_regressor.predict(X_test)
    print(f"XGBoostRegressor : {y_pred}")

    # 3 GradientBoosting
    gradientBoosting_regressor = GradientBoostingRegressor(n_estimators=50, learning_rate=1)
    gradientBoosting_regressor.fit(X_train, y_train)
    y_pred = gradientBoosting_regressor.predict(X_test)
    print(f"GradientBoostingRegressor : {y_pred}")
