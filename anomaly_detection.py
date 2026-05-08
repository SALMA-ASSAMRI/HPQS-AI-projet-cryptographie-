# ============================================================
# anomaly_detection.py — Module 3
# Projet HPQS-AI — Détection d'anomalies avec Isolation Forest
# Salma A : chargement, features, scaler, entraînement, sauvegarde
# Nabila B : prédiction, anomaly_score, affichage anomalies, predict_anomaly()
# ============================================================

import os
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest


# ============================================================
# CONFIGURATION
# ============================================================

METRICS_FILE = "metrics.csv"
MODEL_FILE = "isolation_forest_model.pkl"
SCALER_FILE = "scaler.pkl"

FEATURES = [
    "keygen_time",
    "encrypt_time",
    "decrypt_time",
    "memory_usage",
    "ciphertext_size"
]


# ============================================================
# PARTIE A — CHARGER metrics.csv
# ============================================================

def load_metrics(filepath=METRICS_FILE):
    """
    Charge le fichier metrics.csv avec pandas.
    """

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Fichier introuvable : {filepath}")

    df = pd.read_csv(filepath)

    print("Fichier metrics.csv chargé avec succès")
    print(f"Nombre de sessions : {len(df)}")

    return df


# ============================================================
# PARTIE A — PRÉPARER LES FEATURES
# ============================================================

def prepare_features(df):
    """
    Sélectionne les colonnes numériques utilisées par Isolation Forest.
    """

    missing_columns = [col for col in FEATURES if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Colonnes manquantes dans metrics.csv : {missing_columns}")

    X = df[FEATURES]

    return X


# ============================================================
# PARTIE A — ENTRAÎNER LE MODÈLE
# ============================================================

def train_isolation_forest(df):
    """
    Entraîne un modèle Isolation Forest sur les métriques.
    """

    X = prepare_features(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        contamination=0.1,
        random_state=42
    )

    model.fit(X_scaled)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)

    print("Modèle Isolation Forest entraîné et sauvegardé")
    print(f"Modèle : {MODEL_FILE}")
    print(f"Scaler : {SCALER_FILE}")

    return model, scaler


# ============================================================
# PARTIE B — CHARGER LE MODÈLE ET LE SCALER
# ============================================================

def load_model_and_scaler():
    """
    Personne B :
    Charge le modèle Isolation Forest et le scaler sauvegardés.
    Si les fichiers n'existent pas, le modèle est entraîné automatiquement.
    """

    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)

        print("Modèle et scaler chargés avec succès")

        return model, scaler

    print("Modèle non trouvé. Entraînement automatique...")

    df = load_metrics(METRICS_FILE)
    model, scaler = train_isolation_forest(df)

    return model, scaler


# ============================================================
# PARTIE B — DÉTECTION DES ANOMALIES
# ============================================================

def detect_anomalies(df, model, scaler):
    """
    Personne B :
    Prédit les anomalies sur toutes les sessions de metrics.csv.

    Isolation Forest retourne :
    1  = session normale
    -1 = anomalie
    """

    X = prepare_features(df)
    X_scaled = scaler.transform(X)

    predictions = model.predict(X_scaled)
    scores = model.decision_function(X_scaled)

    df["anomaly_prediction"] = predictions
    df["anomaly_score"] = scores

    df["ai_result"] = df["anomaly_prediction"].apply(
        lambda x: "Anomalie" if x == -1 else "Normal"
    )

    return df


# ============================================================
# PARTIE B — AFFICHER LES ANOMALIES
# ============================================================

def show_anomalies(df):
    """
    Personne B :
    Affiche les lignes détectées comme anomalies.

    Nouvelle modification :
    On affiche aussi anomaly_prediction pour voir la valeur brute du modèle :
    -1 = Anomalie
     1 = Normal
    """

    anomalies = df[df["anomaly_prediction"] == -1]

    print("\nLignes détectées comme anomalies :")

    if anomalies.empty:
        print("Aucune anomalie détectée.")
    else:
        columns_to_show = FEATURES + [
            "kem_level",
            "rsa_level",
            "status",
            "anomaly_prediction",
            "anomaly_score",
            "ai_result"
        ]

        existing_columns = [
            col for col in columns_to_show if col in anomalies.columns
        ]

        print(anomalies[existing_columns].to_string(index=False))

    return anomalies


# ============================================================
# PARTIE B — RÉSUMÉ DES ANOMALIES
# ============================================================

def anomaly_summary(df):
    """
    Personne B :
    Calcule et affiche le nombre d'anomalies par rapport au total.
    """

    total_sessions = len(df)
    anomaly_count = (df["anomaly_prediction"] == -1).sum()
    normal_count = (df["anomaly_prediction"] == 1).sum()

    print("\nRésumé de détection IA :")
    print(f"Total sessions   : {total_sessions}")
    print(f"Sessions normales: {normal_count}")
    print(f"Anomalies        : {anomaly_count}")

    if total_sessions > 0:
        percentage = round((anomaly_count / total_sessions) * 100, 2)
        print(f"Taux d'anomalies : {percentage}%")

    return anomaly_count, total_sessions


# ============================================================
# PARTIE B — PRÉDICTION SUR UNE NOUVELLE SESSION
# ============================================================

def predict_anomaly(session_data):
    """
    Personne B :
    Prédit si une nouvelle session est normale ou anormale.

    session_data doit contenir :
    - keygen_time
    - encrypt_time
    - decrypt_time
    - memory_usage
    - ciphertext_size

    Retourne :
    "Normal" ou "Anomalie"
    """

    model, scaler = load_model_and_scaler()

    session_df = pd.DataFrame([session_data])

    missing_columns = [col for col in FEATURES if col not in session_df.columns]

    if missing_columns:
        raise ValueError(f"Champs manquants dans session_data : {missing_columns}")

    X = session_df[FEATURES]
    X_scaled = scaler.transform(X)

    prediction = model.predict(X_scaled)[0]
    score = model.decision_function(X_scaled)[0]

    result = "Anomalie" if prediction == -1 else "Normal"

    print("\nTest d'une nouvelle session :")
    print(session_data)
    print(f"Anomaly prediction : {prediction}")
    print(f"Résultat IA        : {result}")
    print(f"Anomaly score      : {round(score, 6)}")

    return result


# ============================================================
# PARTIE B — TESTS AVEC VALEURS SIMULÉES
# ============================================================

def test_extreme_values():
    """
    Personne B :
    Teste predict_anomaly() avec :
    - une session normale simulée
    - une session extrême simulée
    """

    normal_session = {
        "keygen_time": 0.10,
        "encrypt_time": 0.005,
        "decrypt_time": 0.003,
        "memory_usage": 110.0,
        "ciphertext_size": 52
    }

    extreme_session = {
        "keygen_time": 4.50,
        "encrypt_time": 2.20,
        "decrypt_time": 1.70,
        "memory_usage": 1400.0,
        "ciphertext_size": 12000
    }

    print("\nTest avec session normale simulée")
    predict_anomaly(normal_session)

    print("\nTest avec session extrême simulée")
    predict_anomaly(extreme_session)


# ============================================================
# EXÉCUTION PRINCIPALE
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("MODULE 3 — DÉTECTION D'ANOMALIES")
    print("Isolation Forest — Salma A + Nabila B")
    print("=" * 60)

    # 1. Charger les métriques
    df = load_metrics(METRICS_FILE)

    # 2. Charger ou entraîner le modèle + scaler
    model, scaler = load_model_and_scaler()

    # 3. Détecter les anomalies sur metrics.csv
    df_result = detect_anomalies(df, model, scaler)

    # 4. Afficher les anomalies avec anomaly_prediction
    show_anomalies(df_result)

    # 5. Afficher le résumé
    anomaly_summary(df_result)

    # 6. Sauvegarder le CSV enrichi
    df_result.to_csv("metrics_with_anomalies.csv", index=False)
    print("\nFichier sauvegardé : metrics_with_anomalies.csv")

    # 7. Tester avec des valeurs simulées
    test_extreme_values()

    print("\nMODULE 3 COMPLET : OK")