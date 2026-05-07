# ============================================================
# optimizer_model.py — Module 4 (Coéquipier A)
# Projet HPQS-AI — Optimisation intelligente (Decision Tree)
# ============================================================

import pandas as pd
import numpy as np
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


# ─────────────────────────────────────────
# ÉTAPE 1 — Créer le dataset de scénarios
# ─────────────────────────────────────────
def creer_dataset():
    data = {
        "app_type": [
            # IoT — léger, peu de mémoire
            "IoT", "IoT", "IoT", "IoT", "IoT",
            "IoT", "IoT", "IoT", "IoT", "IoT",
            # Mobile — léger à moyen
            "Mobile", "Mobile", "Mobile", "Mobile", "Mobile",
            "Mobile", "Mobile", "Mobile", "Mobile", "Mobile",
            # Web — moyen
            "Web", "Web", "Web", "Web", "Web",
            "Web", "Web", "Web", "Web", "Web",
            # Cloud — moyen à élevé
            "Cloud", "Cloud", "Cloud", "Cloud", "Cloud",
            "Cloud", "Cloud", "Cloud", "Cloud", "Cloud",
            # Banque — maximum sécurité
            "Banque", "Banque", "Banque", "Banque", "Banque",
            "Banque", "Banque", "Banque", "Banque", "Banque",
        ],
        "priority": [
            # IoT
            "low","low","low","low","low",
            "low","medium","medium","low","low",
            # Mobile
            "low","low","medium","medium","low",
            "low","medium","low","low","medium",
            # Web
            "medium","medium","medium","high","medium",
            "medium","high","medium","medium","high",
            # Cloud
            "high","high","medium","high","high",
            "medium","high","high","medium","high",
            # Banque
            "high","high","high","high","high",
            "high","high","high","high","high",
        ],
        "memory_mb": [
            # IoT
            256,128,512,256,128,
            256,512,256,128,256,
            # Mobile
            512,256,1024,512,256,
            512,1024,256,512,1024,
            # Web
            1024,2048,1024,2048,1024,
            2048,4096,1024,2048,4096,
            # Cloud
            4096,8192,2048,4096,8192,
            2048,4096,8192,2048,4096,
            # Banque
            4096,8192,4096,8192,4096,
            8192,4096,8192,4096,8192,
        ],
        "kem_level": [
            # IoT → ML-KEM-512
            "ML-KEM-512","ML-KEM-512","ML-KEM-512","ML-KEM-512","ML-KEM-512",
            "ML-KEM-512","ML-KEM-512","ML-KEM-512","ML-KEM-512","ML-KEM-512",
            # Mobile → ML-KEM-512
            "ML-KEM-512","ML-KEM-512","ML-KEM-512","ML-KEM-512","ML-KEM-512",
            "ML-KEM-512","ML-KEM-768","ML-KEM-512","ML-KEM-512","ML-KEM-768",
            # Web → ML-KEM-768
            "ML-KEM-768","ML-KEM-768","ML-KEM-768","ML-KEM-768","ML-KEM-768",
            "ML-KEM-768","ML-KEM-768","ML-KEM-768","ML-KEM-768","ML-KEM-1024",
            # Cloud → ML-KEM-768 / 1024
            "ML-KEM-1024","ML-KEM-1024","ML-KEM-768","ML-KEM-1024","ML-KEM-1024",
            "ML-KEM-768","ML-KEM-1024","ML-KEM-1024","ML-KEM-768","ML-KEM-1024",
            # Banque → ML-KEM-1024
            "ML-KEM-1024","ML-KEM-1024","ML-KEM-1024","ML-KEM-1024","ML-KEM-1024",
            "ML-KEM-1024","ML-KEM-1024","ML-KEM-1024","ML-KEM-1024","ML-KEM-1024",
        ]
    }
    df = pd.DataFrame(data)
    print(f"✅ Dataset créé : {df.shape[0]} scénarios")
    print(df["kem_level"].value_counts())
    return df


# ─────────────────────────────────────────
# ÉTAPE 2 — Encoder les features
# ─────────────────────────────────────────
def encoder_features(df):
    le_app      = LabelEncoder()
    le_priority = LabelEncoder()
    le_kem      = LabelEncoder()

    df["app_type_enc"] = le_app.fit_transform(df["app_type"])
    df["priority_enc"] = le_priority.fit_transform(df["priority"])
    df["kem_level_enc"] = le_kem.fit_transform(df["kem_level"])

    print("✅ Encodage LabelEncoder terminé")
    print(f"   app_type  : {list(le_app.classes_)}")
    print(f"   priority  : {list(le_priority.classes_)}")
    print(f"   kem_level : {list(le_kem.classes_)}")

    return df, le_app, le_priority, le_kem


# ─────────────────────────────────────────
# ÉTAPE 3 — Diviser train/test 80/20
# ─────────────────────────────────────────
def diviser_donnees(df):
    X = df[["app_type_enc", "priority_enc", "memory_mb"]]
    y = df["kem_level_enc"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )
    print(f"✅ Train : {len(X_train)} | Test : {len(X_test)}")
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────
# ÉTAPE 4 — Entraîner Decision Tree
# ─────────────────────────────────────────
def entrainer_decision_tree(X_train, y_train):
    dt = DecisionTreeClassifier(
        max_depth=4,
        random_state=42
    )
    dt.fit(X_train, y_train)
    print("✅ Decision Tree entraîné")
    return dt


# ─────────────────────────────────────────
# ÉTAPE 5 — Entraîner Random Forest
# ─────────────────────────────────────────
def entrainer_random_forest(X_train, y_train):
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    rf.fit(X_train, y_train)
    print("✅ Random Forest entraîné")
    return rf


# ─────────────────────────────────────────
# ÉTAPE 6 — Comparer les deux modèles
# ─────────────────────────────────────────
def comparer_modeles(dt, rf, X_test, y_test):
    # Prédictions
    y_pred_dt = dt.predict(X_test)
    y_pred_rf = rf.predict(X_test)

    # Accuracy
    acc_dt = accuracy_score(y_test, y_pred_dt)
    acc_rf = accuracy_score(y_test, y_pred_rf)

    print("\n📊 Comparaison des modèles :")
    print(f"   Decision Tree accuracy  : {acc_dt:.2f}")
    print(f"   Random Forest accuracy  : {acc_rf:.2f}")

    # Matrice de confusion
    print("\n📊 Matrice de confusion — Decision Tree :")
    print(confusion_matrix(y_test, y_pred_dt))

    print("\n📊 Matrice de confusion — Random Forest :")
    print(confusion_matrix(y_test, y_pred_rf))

    # Choisir le meilleur
    meilleur = "Decision Tree" if acc_dt >= acc_rf else "Random Forest"
    meilleur_model = dt if acc_dt >= acc_rf else rf
    print(f"\n🏆 Meilleur modèle : {meilleur} ({max(acc_dt, acc_rf):.2f})")

    return meilleur_model, meilleur


# ─────────────────────────────────────────
# ÉTAPE 7 — Sauvegarder le meilleur modèle
# ─────────────────────────────────────────
def sauvegarder_modele(model, le_app, le_priority, le_kem):
    joblib.dump(model,      "optimizer_model.pkl")
    joblib.dump(le_app,     "le_app.pkl")
    joblib.dump(le_priority,"le_priority.pkl")
    joblib.dump(le_kem,     "le_kem.pkl")
    print("✅ Modèle et encodeurs sauvegardés")


# ─────────────────────────────────────────
# TEST
# ─────────────────────────────────────────
if __name__ == "__main__":

    # 1. Créer dataset
    df = creer_dataset()

    # 2. Encoder
    df, le_app, le_priority, le_kem = encoder_features(df)

    # 3. Diviser
    X_train, X_test, y_train, y_test = diviser_donnees(df)

    # 4. Entraîner Decision Tree
    dt = entrainer_decision_tree(X_train, y_train)

    # 5. Entraîner Random Forest
    rf = entrainer_random_forest(X_train, y_train)

    # 6. Comparer
    meilleur_model, nom = comparer_modeles(dt, rf, X_test, y_test)

    # 7. Sauvegarder
    sauvegarder_modele(meilleur_model, le_app, le_priority, le_kem)

   
    