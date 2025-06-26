import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

if __name__ == '__main__':
    dfX, sy = load_breast_cancer(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(dfX, sy, test_size=0.2, random_state=42)

    # Rééquilibrage des données
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Créer et entraîner le modèle de RandomForestClassifier
    rfc_model = RandomForestClassifier(random_state=42)
    rfc_model.fit(X_train_resampled, y_train_resampled)

    y_pred = rfc_model.predict(X_test)
    print(f"y_pred = {y_pred}")

    # Matrice de confusion
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"conf_matrix = {conf_matrix}")

    # Rapport de classification
    class_report = classification_report(y_test, y_pred, target_names=load_breast_cancer().target_names)
    print(f"class_report = {class_report}")

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"accuracy = {accuracy:.2f}")