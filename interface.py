import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer


# =========================
# Fonctions de support (Ton code de traitement)
# =========================
def extract_time_features(X):
    dt = pd.to_datetime(X.iloc[:, 0])
    return pd.concat([dt.dt.hour, dt.dt.month], axis=1).values

# Configuration de la page
st.set_page_config(page_title="IA Fou de Bassan - Lab", layout="wide")

st.title("Défi 2 - Fou de Bassan")

# Système d'onglets pour séparer l'entraînement de la prédiction
tab1, tab2 = st.tabs(["Entraînement", "Prédiction"])

MODEL_PATH = "model.joblib"

# =========================
# ONGLET 1 : ENTRAÎNEMENT
# =========================
with tab1:
    st.header("Entraîner le modèle")
    st.info("Déposez votre fichier `train.csv` pour créer le cerveau de l'IA.")
    
    train_file = st.file_uploader("Fichier d'entraînement (CSV)", type="csv", key="train_up")
    
    if train_file is not None:
        df_train = pd.read_csv(train_file)
        st.write("Aperçu des données d'entraînement :", df_train.head(3))
        
        if st.button("🛠️ Lancer l'entraînement"):
            with st.spinner("L'IA analyse les comportements..."):
                features = ['Speed', 'Sinuosity', 'FlightDistance', 'Location', 'LocalDateTime']
                target = 'Behavior'
                
                df_clean = df_train.dropna(subset=features + [target]).copy()
                X = df_clean[features]
                y = df_clean[target]
                
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ['Location']), 
                        ('time', FunctionTransformer(extract_time_features), ['LocalDateTime']) 
                    ],
                    remainder='passthrough'
                )

                model = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('classifier', RandomForestClassifier(
                        n_estimators=500,      
                        max_depth=20,             
                        min_samples_leaf=2,         
                        max_features='sqrt',       
                        class_weight='balanced_subsample', 
                        random_state=42,
                        n_jobs=-1
                    ))
                ])
                
                model.fit(X, y)
                joblib.dump(model, MODEL_PATH)
                st.success("Modèle entraîné et sauvegardé sous 'model.joblib' !")

# =========================
# ONGLET 2 : PRÉDICTION
# =========================
with tab2:
    st.header("Prédire de nouveaux comportements")
    
    if not os.path.exists(MODEL_PATH):
        st.warning("Aucun modèle trouvé. Veuillez d'abord passer par l'onglet 'Entraînement'.")
    else:
        model = joblib.load(MODEL_PATH)
        st.success("Modèle prêt à l'emploi.")
        
        test_file = st.file_uploader("Fichier à prédire (CSV)", type="csv", key="test_up")
        
        if test_file is not None:
            df_test = pd.read_csv(test_file)
            required_cols = ['Speed', 'Sinuosity', 'FlightDistance', 'Location', 'LocalDateTime']
            
            if all(col in df_test.columns for col in required_cols):
                if st.button("Prédire"):
                    preds = model.predict(df_test[required_cols])
                    df_test['Behavior_Predicted'] = preds
                    
                    st.divider()
                    
                    # Affichage des métriques si la colonne réelle existe
                    if 'Behavior' in df_test.columns:
                        acc = accuracy_score(df_test['Behavior'], preds)
                        st.metric("Précision de l'IA", f"{acc:.2%}")
                    
                    # Graphique de répartition
                    st.subheader("Répartition des comportements prédits")
                    st.bar_chart(df_test['Behavior_Predicted'].value_counts())
                    
                    # Aperçu et téléchargement
                    st.write("Résultats (20 premières lignes) :")
                    st.dataframe(df_test[['LocalDateTime', 'Behavior_Predicted']].head(20))
                    
                    csv = df_test.to_csv(index=False).encode('utf-8')
                    st.download_button("Télécharger les prédictions", csv, "resultats.csv", "text/csv")
            else:
                st.error("Colonnes manquantes dans le fichier test.")