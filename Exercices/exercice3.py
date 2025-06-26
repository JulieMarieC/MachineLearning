import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def custom_score(y_test, y_pred, fn_value=5, fp_value=1, tp_value=0, tn_value=0):
    tp, tn, fp, fn = confusion_matrix(y_test, y_pred).ravel()
    J = tp*tp_value + tn*tn_value + fp*fp_value + fn*fn_value
    return J

if __name__ == '__main__':
    dfX, sy = load_breast_cancer(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(dfX, sy, test_size=0.2, random_state=42)

    adaboost_regressor = AdaBoostClassifier(n_estimators=50, learning_rate=1)
    adaboost_regressor.fit(X_train, y_train)

    # Faire des prédictions sur l'ensemble de test
    y_pred = adaboost_regressor.predict(X_test)
    print(f"y_pred : {y_pred}")

    score = custom_score(y_test, y_pred)
    print(f"score : {score}")

