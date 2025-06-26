import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    dfX, sy = load_breast_cancer(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(dfX, sy, test_size=0.2, random_state=42)

    # Créer et entraîner le modèle de RandomForestClassifier
    rfc_model = RandomForestClassifier(random_state=42)
    rfc_model.fit(X_train, y_train)

    y_pred = rfc_model.predict(X_test)
    print(f"y_pred = {y_pred}")

