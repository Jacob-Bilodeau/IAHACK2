# =========================
# Importation
# =========================
import os
import json
import argparse
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# =========================
# TRAIN
# =========================
def train_model(train_csv, model_dir):
    df = pd.read_csv(train_csv)

    features = ['Speed', 'Sinuosity', 'FlightDistance']
    target = 'Behavior'
    

    df_clean = df.dropna(subset=features + [target]).copy()

    df_final = df_clean[features + [target]]
    os.makedirs(model_dir, exist_ok=True)
    filtered_path = os.path.join(model_dir, 'train_filtered.csv')
    df_final.to_csv(filtered_path, index=False)

    X = df_clean[features]
    y = df_clean[target]
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    joblib.dump(model, os.path.join(model_dir, 'model.joblib'))
    print('Modèle entraîné sur les 3 caractéristiques de base.')

# =========================
#  LOAD
# =========================
def load_model(model_dir):
    path = os.path.join(model_dir, 'model.joblib')
    if not os.path.exists(path):
        raise FileNotFoundError('Modèle non trouvé')
    return joblib.load(path)

# =========================
# EVALUATE
# =========================
def evaluate_model(model, test_csv, output_json):

    df = pd.read_csv(test_csv)

    y_true = df['Behavior']
    y_pred = model.predict(df[['Speed', 'Sinuosity', 'FlightDistance']])

   #  Metrics 
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = confusion_matrix(y_true, y_pred, normalize='true')
    diag_mean = cm_normalized.diagonal().mean()

    metrics = {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "diag_mean": float(diag_mean)
    }

    with open(output_json, "w") as f:
        json.dump(metrics, f, indent=4)

    print("Évaluation terminée")

# =========================
#  MAIN
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline")

    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("train_csv")
    train_parser.add_argument("model_dir")

    test_parser = subparsers.add_parser("test")
    test_parser.add_argument("test_csv")
    test_parser.add_argument("model_dir")
    test_parser.add_argument("output_json")

    args = parser.parse_args()

    if args.command == "train":
        train_model(args.train_csv, args.model_dir)

    elif args.command == "test":
        model = load_model(args.model_dir)
        evaluate_model(model, args.test_csv, args.output_json)

    else:
        parser.print_help()