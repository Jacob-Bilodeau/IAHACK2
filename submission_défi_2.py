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
# Préparation des données
# =========================
def prepare_data(df):
    # Date -> Heure
    df['Hour'] = pd.to_datetime(df['LocalDateTime']).dt.hour
    
    # sea = 0, colony = 1
    df['Is_Colony'] = df['Location'].map({'sea': 0, 'colony': 1})
    
    features = ['Speed', 'Sinuosity', 'FlightDistance', 'Hour', 'Is_Colony']
    target = 'Behavior'
    
    return df, features, target

# =========================
# TRAIN
# =========================
def train_model(train_csv, model_dir):
    print("Chargement et filtrage des données...")
    df = pd.read_csv(train_csv)

    df, features, target = prepare_data(df)
    
    # Filtre
    df_clean = df.dropna(subset=features + [target]).copy()
    
    df_final = df_clean[features + [target]]
    
    os.makedirs(model_dir, exist_ok=True)
    filtered_path = os.path.join(model_dir, "train_filtered.csv")
    df_final.to_csv(filtered_path, index=False)
    print(f"Fichier filtré (colonnes et lignes) créé : {filtered_path}")

    # Entraînement
    X = df_clean[features]
    y = df_clean[target]
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    joblib.dump(model, os.path.join(model_dir, "model.joblib"))
    print("Modèle entraîné.")

# =========================
#  LOAD
# =========================
def load_model(model_dir):
    path = os.path.join(model_dir, "model.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError("Modèle non trouvé")
    return joblib.load(path)

# =========================
# EVALUATE
# =========================
def evaluate_model(model, test_csv, output_json):
    print("Évaluation en cours...")
    df = pd.read_csv(test_csv)

    df, features, target = prepare_data(df)
    
    df_clean = df.dropna(subset=features + [target]).copy()
    
    X_test = df_clean[features]
    y_true = df_clean[target]
    
    y_pred = model.predict(X_test)

    results_df = df_clean[features + [target]].copy() 
    results_df['Predicted_Behavior'] = y_pred
    results_df['Is_Correct'] = (results_df[target] == results_df['Predicted_Behavior'])

    pred_path = os.path.join(os.path.dirname(output_json), "predictions_comparaison.csv")
    results_df.to_csv(pred_path, index=False)
    print(f"Tableau comparatif sauvegardé : {pred_path}")

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    
    cm_normalized = confusion_matrix(y_true, y_pred, normalize='true')

    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Greens', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title(f'Matrice de Confusion (Accuracy: {acc:.2f})')
    plt.xlabel('Prédictions de l\'IA')
    plt.ylabel('Réalité (Terrain)')
    
    img_path = output_json.replace('.json', '_matrix.png')
    plt.savefig(img_path)
    print(f"Matrice de confusion sauvegardée : {img_path}")

    metrics = {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "diag_mean": float(cm_normalized.diagonal().mean())
    }

    with open(output_json, "w") as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Évaluation terminée. Score global : {acc:.2%}")
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