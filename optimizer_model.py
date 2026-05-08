# ============================================================
# optimizer_model.py — Module 4
# Projet HPQS-AI — Optimisation intelligente du niveau ML-KEM
#
# Salma A :
# - créer le dataset
# - encoder les colonnes texte
# - diviser train/test
# - entraîner Decision Tree et Random Forest
#
# Nabila B :
# - choisir le meilleur modèle
# - sauvegarder le modèle et les encodeurs
# - créer recommend_kem()
# - tester 5 scénarios
# - générer un rapport texte
# ============================================================

import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# ============================================================
# NOMS DES FICHIERS SAUVEGARDÉS
# ============================================================

MODEL_FILE = "optimizer_model.pkl"
LE_APP_FILE = "le_app.pkl"
LE_PRIORITY_FILE = "le_priority.pkl"
LE_KEM_FILE = "le_kem.pkl"
REPORT_FILE = "optimizer_report.txt"


# ============================================================
# 1. CRÉATION DU DATASET
# ============================================================

def creer_dataset():
    """
    Crée un dataset de scénarios.
    Chaque ligne représente une situation d'utilisation.
    Le modèle apprend quel niveau ML-KEM recommander.
    """

    scenarios = [
        # IoT : appareils faibles → ML-KEM-512
        ["IoT", "low", 128, "ML-KEM-512"],
        ["IoT", "low", 256, "ML-KEM-512"],
        ["IoT", "low", 512, "ML-KEM-512"],
        ["IoT", "medium", 256, "ML-KEM-512"],
        ["IoT", "medium", 512, "ML-KEM-512"],
        ["IoT", "low", 128, "ML-KEM-512"],
        ["IoT", "low", 256, "ML-KEM-512"],
        ["IoT", "medium", 512, "ML-KEM-512"],
        ["IoT", "low", 256, "ML-KEM-512"],
        ["IoT", "medium", 1024, "ML-KEM-768"],

        # Mobile : compromis → 512 ou 768
        ["Mobile", "low", 256, "ML-KEM-512"],
        ["Mobile", "low", 512, "ML-KEM-512"],
        ["Mobile", "medium", 512, "ML-KEM-768"],
        ["Mobile", "medium", 1024, "ML-KEM-768"],
        ["Mobile", "high", 1024, "ML-KEM-768"],
        ["Mobile", "low", 512, "ML-KEM-512"],
        ["Mobile", "medium", 2048, "ML-KEM-768"],
        ["Mobile", "high", 2048, "ML-KEM-768"],
        ["Mobile", "medium", 1024, "ML-KEM-768"],
        ["Mobile", "low", 256, "ML-KEM-512"],

        # Web : usage général → ML-KEM-768
        ["Web", "low", 512, "ML-KEM-512"],
        ["Web", "medium", 1024, "ML-KEM-768"],
        ["Web", "medium", 2048, "ML-KEM-768"],
        ["Web", "high", 2048, "ML-KEM-768"],
        ["Web", "high", 4096, "ML-KEM-1024"],
        ["Web", "medium", 1024, "ML-KEM-768"],
        ["Web", "medium", 2048, "ML-KEM-768"],
        ["Web", "high", 4096, "ML-KEM-1024"],
        ["Web", "low", 1024, "ML-KEM-768"],
        ["Web", "medium", 4096, "ML-KEM-768"],

        # Cloud : ressources élevées → 768 ou 1024
        ["Cloud", "medium", 2048, "ML-KEM-768"],
        ["Cloud", "medium", 4096, "ML-KEM-768"],
        ["Cloud", "high", 4096, "ML-KEM-1024"],
        ["Cloud", "high", 8192, "ML-KEM-1024"],
        ["Cloud", "high", 16384, "ML-KEM-1024"],
        ["Cloud", "medium", 4096, "ML-KEM-768"],
        ["Cloud", "high", 8192, "ML-KEM-1024"],
        ["Cloud", "medium", 2048, "ML-KEM-768"],
        ["Cloud", "high", 4096, "ML-KEM-1024"],
        ["Cloud", "high", 8192, "ML-KEM-1024"],

        # Banque : sécurité maximale → ML-KEM-1024
        ["Banque", "high", 2048, "ML-KEM-1024"],
        ["Banque", "high", 4096, "ML-KEM-1024"],
        ["Banque", "high", 8192, "ML-KEM-1024"],
        ["Banque", "medium", 4096, "ML-KEM-1024"],
        ["Banque", "high", 16384, "ML-KEM-1024"],
        ["Banque", "high", 4096, "ML-KEM-1024"],
        ["Banque", "high", 8192, "ML-KEM-1024"],
        ["Banque", "medium", 8192, "ML-KEM-1024"],
        ["Banque", "high", 2048, "ML-KEM-1024"],
        ["Banque", "high", 4096, "ML-KEM-1024"],
    ]

    df = pd.DataFrame(
        scenarios,
        columns=["app_type", "priority", "memory_mb", "kem_level"]
    )

    print("Dataset créé :", len(df), "scénarios")
    print("\nRépartition des niveaux ML-KEM :")
    print(df["kem_level"].value_counts())

    return df


# ============================================================
# 2. ENCODAGE DES DONNÉES TEXTE
# ============================================================

def encoder_features(df):
    """
    Convertit les textes en nombres.
    Exemple :
    IoT, Web, Banque → 0, 1, 2...
    low, medium, high → 0, 1, 2...
    """

    le_app = LabelEncoder()
    le_priority = LabelEncoder()
    le_kem = LabelEncoder()

    df["app_type_enc"] = le_app.fit_transform(df["app_type"])
    df["priority_enc"] = le_priority.fit_transform(df["priority"])
    df["kem_level_enc"] = le_kem.fit_transform(df["kem_level"])

    print("\nEncodage terminé")
    print("Applications :", list(le_app.classes_))
    print("Priorités    :", list(le_priority.classes_))
    print("Niveaux KEM  :", list(le_kem.classes_))

    return df, le_app, le_priority, le_kem


# ============================================================
# 3. DIVISION TRAIN / TEST
# ============================================================

def diviser_donnees(df):
    """
    Divise les données :
    - 80% pour entraîner
    - 20% pour tester
    """

    X = df[["app_type_enc", "priority_enc", "memory_mb"]]
    y = df["kem_level_enc"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nDonnées divisées")
    print("Train :", len(X_train))
    print("Test  :", len(X_test))

    return X_train, X_test, y_train, y_test


# ============================================================
# 4. ENTRAÎNER DECISION TREE
# ============================================================

def entrainer_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(
        max_depth=4,
        random_state=42
    )

    model.fit(X_train, y_train)

    print("Decision Tree entraîné")

    return model


# ============================================================
# 5. ENTRAÎNER RANDOM FOREST
# ============================================================

def entrainer_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    print("Random Forest entraîné")

    return model


# ============================================================
# 6. COMPARER LES MODÈLES
# ============================================================

def comparer_modeles(decision_tree, random_forest, X_test, y_test):
    """
    Compare les deux modèles avec accuracy.
    Le meilleur est celui qui a la meilleure accuracy.
    """

    pred_dt = decision_tree.predict(X_test)
    pred_rf = random_forest.predict(X_test)

    acc_dt = accuracy_score(y_test, pred_dt)
    acc_rf = accuracy_score(y_test, pred_rf)

    print("\nComparaison des modèles :")
    print("Decision Tree accuracy :", round(acc_dt, 2))
    print("Random Forest accuracy :", round(acc_rf, 2))

    print("\nMatrice de confusion — Decision Tree :")
    print(confusion_matrix(y_test, pred_dt))

    print("\nMatrice de confusion — Random Forest :")
    print(confusion_matrix(y_test, pred_rf))

    if acc_dt >= acc_rf:
        best_model = decision_tree
        best_name = "Decision Tree"
        best_accuracy = acc_dt
    else:
        best_model = random_forest
        best_name = "Random Forest"
        best_accuracy = acc_rf

    print("\nMeilleur modèle :", best_name, "accuracy =", round(best_accuracy, 2))

    results = {
        "decision_tree_accuracy": acc_dt,
        "random_forest_accuracy": acc_rf,
        "best_model_name": best_name,
        "best_accuracy": best_accuracy
    }

    return best_model, results


# ============================================================
# 7. SAUVEGARDER MODÈLE + ENCODEURS
# ============================================================

def sauvegarder_modele(best_model, le_app, le_priority, le_kem):
    """
    Sauvegarde le modèle et les encodeurs.
    Ces fichiers seront utilisés dans app.py.
    """

    joblib.dump(best_model, MODEL_FILE)
    joblib.dump(le_app, LE_APP_FILE)
    joblib.dump(le_priority, LE_PRIORITY_FILE)
    joblib.dump(le_kem, LE_KEM_FILE)

    print("\nFichiers sauvegardés :")
    print(MODEL_FILE)
    print(LE_APP_FILE)
    print(LE_PRIORITY_FILE)
    print(LE_KEM_FILE)


# ============================================================
# 8. CHARGER MODÈLE + ENCODEURS
# ============================================================

def charger_modele_et_encodeurs():
    model = joblib.load(MODEL_FILE)
    le_app = joblib.load(LE_APP_FILE)
    le_priority = joblib.load(LE_PRIORITY_FILE)
    le_kem = joblib.load(LE_KEM_FILE)

    return model, le_app, le_priority, le_kem


# ============================================================
# 9. NORMALISER LES ENTRÉES UTILISATEUR
# ============================================================

def normaliser_app_type(app_type):
    app = str(app_type).strip().lower()

    mapping = {
        "iot": "IoT",
        "web": "Web",
        "mobile": "Mobile",
        "cloud": "Cloud",
        "banque": "Banque",
        "bank": "Banque"
    }

    if app not in mapping:
        raise ValueError("app_type doit être : IoT, Web, Mobile, Cloud ou Banque")

    return mapping[app]


def normaliser_priority(priority):
    p = str(priority).strip().lower()

    mapping = {
        "low": "low",
        "performance": "low",
        "faible": "low",

        "medium": "medium",
        "equilibre": "medium",
        "équilibre": "medium",
        "moyenne": "medium",

        "high": "high",
        "securite maximale": "high",
        "sécurité maximale": "high",
        "securite": "high",
        "sécurité": "high",
        "haute": "high"
    }

    if p not in mapping:
        raise ValueError("priority doit être : Performance, Équilibre ou Sécurité maximale")

    return mapping[p]


def normaliser_memory(memory):
    """
    Accepte :
    - un nombre : 256, 1024, 4096...
    - ou un texte : Faible, Moyenne, Élevée
    """

    if isinstance(memory, int) or isinstance(memory, float):
        return int(memory)

    m = str(memory).strip().lower()

    mapping = {
        "faible": 256,
        "low": 256,

        "moyenne": 1024,
        "medium": 1024,

        "elevee": 4096,
        "élevée": 4096,
        "haute": 4096,
        "high": 4096
    }

    if m not in mapping:
        raise ValueError("memory doit être : Faible, Moyenne, Élevée ou un nombre en MB")

    return mapping[m]


# ============================================================
# 10. FONCTION PRINCIPALE DE NABILA B
# ============================================================

def recommend_kem(app_type, priority, memory):
    """
    Recommande le niveau ML-KEM optimal.

    Entrées :
    app_type : IoT, Web, Mobile, Cloud, Banque
    priority : Performance, Équilibre, Sécurité maximale
    memory : Faible, Moyenne, Élevée ou nombre en MB

    Sortie :
    ML-KEM-512, ML-KEM-768 ou ML-KEM-1024
    """

    model, le_app, le_priority, le_kem = charger_modele_et_encodeurs()

    app_clean = normaliser_app_type(app_type)
    priority_clean = normaliser_priority(priority)
    memory_mb = normaliser_memory(memory)

    app_encoded = le_app.transform([app_clean])[0]
    priority_encoded = le_priority.transform([priority_clean])[0]

    X_new = pd.DataFrame([{
        "app_type_enc": app_encoded,
        "priority_enc": priority_encoded,
        "memory_mb": memory_mb
    }])

    prediction_encoded = model.predict(X_new)[0]

    kem_recommended = le_kem.inverse_transform([prediction_encoded])[0]

    return kem_recommended


# ============================================================
# 11. TESTER 5 SCÉNARIOS
# ============================================================

def tester_recommandations():
    scenarios = [
        ["IoT", "Performance", "Faible"],
        ["Mobile", "Équilibre", "Moyenne"],
        ["Web", "Équilibre", "Moyenne"],
        ["Cloud", "Sécurité maximale", "Élevée"],
        ["Banque", "Sécurité maximale", "Élevée"],
    ]

    print("\nTests de recommandation :")

    results = []

    for app_type, priority, memory in scenarios:
        kem = recommend_kem(app_type, priority, memory)

        print(app_type, "|", priority, "|", memory, "→", kem)

        results.append({
            "app_type": app_type,
            "priority": priority,
            "memory": memory,
            "recommendation": kem
        })

    return results


# ============================================================
# 12. GÉNÉRER LE RAPPORT TEXTE
# ============================================================

def generer_rapport(best_model, results, test_results):
    """
    Génère un rapport texte avec :
    - accuracy
    - meilleur modèle
    - importance des features
    - tests de recommandation
    """

    feature_names = ["app_type", "priority", "memory_mb"]
    importances = best_model.feature_importances_

    with open(REPORT_FILE, "w", encoding="utf-8") as file:
        file.write("Rapport Module 4 — Optimisation intelligente ML-KEM\n")
        file.write("=" * 60 + "\n\n")

        file.write("Objectif :\n")
        file.write(
            "Le but du Module 4 est de recommander automatiquement "
            "le meilleur niveau ML-KEM selon le contexte d'utilisation.\n\n"
        )

        file.write("Meilleur modèle :\n")
        file.write(results["best_model_name"] + "\n\n")

        file.write("Accuracy :\n")
        file.write(f"Decision Tree accuracy : {results['decision_tree_accuracy']:.2f}\n")
        file.write(f"Random Forest accuracy : {results['random_forest_accuracy']:.2f}\n")
        file.write(f"Best accuracy : {results['best_accuracy']:.2f}\n\n")

        file.write("Feature importance :\n")
        for name, importance in zip(feature_names, importances):
            file.write(f"{name} : {importance:.4f}\n")

        file.write("\nTests de recommandation :\n")
        for item in test_results:
            file.write(
                f"{item['app_type']} | {item['priority']} | {item['memory']} "
                f"→ {item['recommendation']}\n"
            )

    print("\nRapport généré :", REPORT_FILE)


# ============================================================
# 13. PIPELINE COMPLET DU MODULE 4
# ============================================================

def train_and_save_optimizer():
    df = creer_dataset()

    df, le_app, le_priority, le_kem = encoder_features(df)

    X_train, X_test, y_train, y_test = diviser_donnees(df)

    decision_tree = entrainer_decision_tree(X_train, y_train)

    random_forest = entrainer_random_forest(X_train, y_train)

    best_model, results = comparer_modeles(
        decision_tree,
        random_forest,
        X_test,
        y_test
    )

    sauvegarder_modele(
        best_model,
        le_app,
        le_priority,
        le_kem
    )

    test_results = tester_recommandations()

    generer_rapport(
        best_model,
        results,
        test_results
    )

    return best_model, results


# ============================================================
# TEST PRINCIPAL
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("MODULE 4 — OPTIMISATION INTELLIGENTE ML-KEM")
    print("Decision Tree / Random Forest — Salma A + Nabila B")
    print("=" * 60)

    train_and_save_optimizer()

    print("\nMODULE 4 COMPLET : OK")