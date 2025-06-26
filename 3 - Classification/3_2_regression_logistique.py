import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression

if __name__ == '__main__':
    dfX, sy = load_breast_cancer(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(dfX, sy, test_size=0.2, random_state=42)

    # Créer et entraîner le modèle de régression logistique
    logistic_model = LogisticRegression(max_iter= 10000, random_state=42)
    logistic_model.fit(X_train, y_train)

    y_pred = logistic_model.predict(X_test)
    print(f"y_pred : {y_pred}")

    