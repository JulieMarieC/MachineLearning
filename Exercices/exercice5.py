import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

if __name__ == '__main__':
    dfX,sy = load_breast_cancer(return_X_y=True,as_frame=True)

    X_train, X_test, y_train, y_test = train_test_split(dfX, sy, test_size=0.3, random_state=42)

    classifiers = {
        'decision_tree': DecisionTreeClassifier(random_state=42),
        'regression_logistique': LogisticRegression(max_iter=10000, random_state=42),
        'knn': KNeighborsClassifier(),
        'support_vector_classifier': SVC(probability=True, random_state=42),
        'random_forest': RandomForestClassifier(random_state=42)
        }

    for name, classifier in classifiers.items():
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', classifier)
        ])

        scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')

        print(f"name: {name}")
        print(f"accuracy: {scores.mean():0.2f}\n")