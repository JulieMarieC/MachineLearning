import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    dfX, sy = load_breast_cancer(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(dfX, sy, test_size=0.2, random_state=42)

    dummy_clf = DummyClassifier(strategy='most_frequent', random_state=42)
    dummy_clf.fit(X_train, y_train)
    y_pred_before = dummy_clf.predict(X_test)
    print(f"y_pred_before : {y_pred_before}")

    print(y_train.value_counts())

    # Equilibrer les données
    # 1 - OverSampling
    # 2 - UnderSampling

    smote = SMOTE()
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    print(f"y_train_balanced : {y_train_balanced.value_counts()}")
    dummy_clf.fit(X_train_balanced, y_train_balanced)
    y_pred_after = dummy_clf.predict(X_test)
    print(f"y_pred_after : {y_pred_after}")
