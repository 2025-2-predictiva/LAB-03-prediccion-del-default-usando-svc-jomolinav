
 

#Version 2

import pandas as pd
import json
import gzip
import pickle
import os

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix


def load_data(file_path):
    return pd.read_csv(file_path, compression='zip')


def clean_data(data):
    data = data.rename(columns={"default payment next month": "default"})
    data = data.drop(columns=["ID"])
    data = data.dropna()
    data["EDUCATION"] = data["EDUCATION"].apply(lambda x: 4 if x > 4 else x)
    return data


def split_data(data):
    X = data.drop(columns=["default"])
    y = data["default"]
    return X, y


def create_pipeline(X):
    numeric = [
        "LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6", 
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]
    categorical = ["SEX", "EDUCATION", "MARRIAGE"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("categoria", OneHotEncoder(handle_unknown='ignore'), categorical),
            ('scaler',StandardScaler(with_mean=True, with_std=True),numeric),
        ])

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('pca', PCA()),
        ('feature_selection', SelectKBest(score_func=f_classif, k=10)), 
        ('classifier', SVC(kernel='rbf', random_state=42))
    ])

    return pipeline


def optimize_hyperparameters(pipeline, X_train, y_train):
    param_grid = {
    "pca__n_components": [20, X_train.shape[1] - 2],
    'feature_selection__k': [12],
    'classifier__kernel': ["rbf"],
    'classifier__gamma': [0.1],
    }

    gridSearch = GridSearchCV(
        pipeline,
        param_grid,
        cv=10,
        scoring='balanced_accuracy',
        n_jobs=-1,
        verbose=2,
        refit=True
    )

    gridSearch.fit(X_train, y_train)
    return gridSearch


def save_model(model):
    os.makedirs("files/models", exist_ok=True)
    with gzip.open("files/models/model.pkl.gz", "wb") as f:
        pickle.dump(model, f)


def save_metrics(model, x_train, y_train, x_test, y_test):
    os.makedirs("files/output", exist_ok=True)

    metrics = []

    # MÉTRICAS TRAIN (posición 0)
    y_train_pred = model.predict(x_train)
    metrics.append({
        "type": "metrics",
        "dataset": "train",
        "precision": float(precision_score(y_train, y_train_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_train, y_train_pred)),
        "recall": float(recall_score(y_train, y_train_pred)),
        "f1_score": float(f1_score(y_train, y_train_pred)),
    })

    # MÉTRICAS TEST (posición 1)
    y_test_pred = model.predict(x_test)
    metrics.append({
        "type": "metrics",
        "dataset": "test",
        "precision": float(precision_score(y_test, y_test_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_test_pred)),
        "recall": float(recall_score(y_test, y_test_pred)),
        "f1_score": float(f1_score(y_test, y_test_pred)),
    })

    # GUARDA SIN GZIP, UNA LÍNEA POR OBJETO
    with open("files/output/metrics.json", "w", encoding="utf-8") as f:
        for m in metrics:
            f.write(json.dumps(m) + "\n")

def save_confusion_matrices(model, x_train, y_train, x_test, y_test):
    metrics = []

    # CARGAR lo que estaba en el archivo
    with open("files/output/metrics.json", "r", encoding="utf-8") as f:
        for line in f:
            metrics.append(json.loads(line))

    # CM TRAIN (posición 2)
    cm_train = confusion_matrix(y_train, model.predict(x_train))
    metrics.append({
        "type": "cm_matrix",
        "dataset": "train",
        "true_0": {
            "predicted_0": int(cm_train[0][0]),
            "predicted_1": int(cm_train[0][1])
        },
        "true_1": {
            "predicted_0": int(cm_train[1][0]),
            "predicted_1": int(cm_train[1][1])
        }
    })

    # CM TEST (posición 3)
    cm_test = confusion_matrix(y_test, model.predict(x_test))
    metrics.append({
        "type": "cm_matrix",
        "dataset": "test",
        "true_0": {
            "predicted_0": int(cm_test[0][0]),
            "predicted_1": int(cm_test[0][1])
        },
        "true_1": {
            "predicted_0": int(cm_test[1][0]),
            "predicted_1": int(cm_test[1][1])
        }
    })

    # GUARDAR TODO DE NUEVO EN JSON NORMAL
    with open("files/output/metrics.json", "w", encoding="utf-8") as f:
        for m in metrics:
            f.write(json.dumps(m) + "\n")

if __name__ == "__main__":
    os.makedirs("files/output", exist_ok=True)

    train = load_data("files/input/train_data.csv.zip")
    test = load_data("files/input/test_data.csv.zip")

    train = clean_data(train)
    test = clean_data(test)

    X_train, y_train = split_data(train)
    X_test, y_test = split_data(test)

    pipeline = create_pipeline(X_train)
    model = optimize_hyperparameters(pipeline, X_train, y_train)

    save_model(model)

    # guardar métricas
    save_metrics(model, X_train, y_train, X_test, y_test)
    save_confusion_matrices(model, X_train, y_train, X_test, y_test)
                     