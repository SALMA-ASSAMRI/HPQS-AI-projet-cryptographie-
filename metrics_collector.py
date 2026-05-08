# ============================================================
# metrics_collector.py — Module 2
# Projet HPQS-AI — Collecte des métriques de performance
# Salma A : keygen_time, encrypt_time, ciphertext_size, kem_level
# Nabila B : decrypt_time, memory_usage, rsa_level, status, anomalies
# ============================================================

import time
import csv
import os
import random
import pandas as pd
import psutil

from crypto_module import (
    generate_rsa_keys,
    generate_mlkem_keys,
    encapsulate_K1,
    hybrid_encrypt,
    hybrid_decrypt
)


# ============================================================
# CHAMPS FINAUX DU FICHIER metrics.csv
# ============================================================

FIELDNAMES = [
    "keygen_time",
    "encrypt_time",
    "decrypt_time",
    "memory_usage",
    "ciphertext_size",
    "kem_level",
    "rsa_level",
    "status"
]


# ============================================================
# MESURE 1 — keygen_time
# ============================================================

def measure_keygen_time(kem_level=512):
    """
    Mesure le temps de génération des clés :
    - RSA-2048
    - ML-KEM-512 / 768 / 1024
    """

    start = time.time()

    rsa_private_key, rsa_public_key = generate_rsa_keys()
    mlkem_public_key, mlkem_secret_key = generate_mlkem_keys(kem_level)

    end = time.time()

    keygen_time = round(end - start, 6)

    return keygen_time, rsa_private_key, rsa_public_key, mlkem_public_key, mlkem_secret_key


# ============================================================
# MESURE 2 — encrypt_time + ciphertext_size
# ============================================================

def measure_encrypt_time(message, rsa_public_key, mlkem_public_key, kem_level=512):
    """
    Mesure le temps de chiffrement complet :
    - encapsulation K1 avec ML-KEM
    - chiffrement du message avec hybrid_encrypt()
    """

    start = time.time()

    ciphertext_kem, K1 = encapsulate_K1(mlkem_public_key, kem_level)

    bundle = hybrid_encrypt(message, rsa_public_key, K1)

    bundle["ciphertext_kem"] = ciphertext_kem
    bundle["kem_level"] = kem_level

    end = time.time()

    encrypt_time = round(end - start, 6)
    ciphertext_size = len(bundle["ciphertext"])

    return encrypt_time, ciphertext_size, bundle


# ============================================================
# MESURE 3 — decrypt_time
# ============================================================

def measure_decrypt_time(bundle, rsa_private_key, mlkem_secret_key, kem_level=512):
    """
    Personne B :
    Mesure le temps de déchiffrement complet avec hybrid_decrypt().
    """

    start = time.time()

    decrypted_message = hybrid_decrypt(
        bundle,
        rsa_private_key,
        mlkem_secret_key,
        kem_level
    )

    end = time.time()

    decrypt_time = round(end - start, 6)

    return decrypt_time, decrypted_message


# ============================================================
# MESURE 4 — memory_usage
# ============================================================

def measure_memory_usage():
    """
    Personne B :
    Mesure la mémoire RAM utilisée par le processus Python en MB.
    """

    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / (1024 * 1024)

    return round(memory_mb, 2)


# ============================================================
# SAUVEGARDE CSV
# ============================================================

def prepare_csv_file(filepath="metrics.csv"):
    """
    Vérifie si metrics.csv existe déjà avec un ancien format.
    Si l'ancien fichier n'a pas les nouvelles colonnes, on le sauvegarde.
    """

    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        try:
            df = pd.read_csv(filepath, nrows=0)

            if list(df.columns) != FIELDNAMES:
                backup_path = filepath.replace(".csv", "_old.csv")
                os.replace(filepath, backup_path)
                print(f"Ancien fichier sauvegardé sous : {backup_path}")

        except Exception:
            backup_path = filepath.replace(".csv", "_old.csv")
            os.replace(filepath, backup_path)
            print(f"Ancien fichier illisible sauvegardé sous : {backup_path}")


def save_metrics(data, filepath="metrics.csv"):
    """
    Enregistre une ligne de métriques dans metrics.csv.
    """

    prepare_csv_file(filepath)

    file_exists = os.path.exists(filepath)

    with open(filepath, mode="a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)

        if not file_exists:
            writer.writeheader()

        writer.writerow(data)

    print(f"Métriques sauvegardées dans {filepath}")


# ============================================================
# SESSION NORMALE
# ============================================================

def collect_normal_session(message, kem_level=512, rsa_level=2048, filepath="metrics.csv"):
    """
    Lance une session cryptographique normale :
    1. Génération des clés
    2. Chiffrement
    3. Déchiffrement
    4. Mesure mémoire
    5. Sauvegarde CSV
    """

    keygen_time, rsa_private_key, rsa_public_key, mlkem_public_key, mlkem_secret_key = measure_keygen_time(
        kem_level
    )

    encrypt_time, ciphertext_size, bundle = measure_encrypt_time(
        message,
        rsa_public_key,
        mlkem_public_key,
        kem_level
    )

    decrypt_time, decrypted_message = measure_decrypt_time(
        bundle,
        rsa_private_key,
        mlkem_secret_key,
        kem_level
    )

    memory_usage = measure_memory_usage()

    if decrypted_message == message:
        status = "normal"
    else:
        status = "error"

    data = {
        "keygen_time": keygen_time,
        "encrypt_time": encrypt_time,
        "decrypt_time": decrypt_time,
        "memory_usage": memory_usage,
        "ciphertext_size": ciphertext_size,
        "kem_level": f"ML-KEM-{kem_level}",
        "rsa_level": rsa_level,
        "status": status
    }

    save_metrics(data, filepath)

    return data


# ============================================================
# ENTRÉES ANORMALES POUR MODULE 3
# ============================================================

def generate_anomalous_entries(filepath="metrics.csv", count=5):
    """
    Personne B :
    Génère des lignes anormales simulées.
    Ces lignes serviront plus tard au Module 3 pour entraîner/tester l'IA.
    """

    for _ in range(count):
        data = {
            "keygen_time": round(random.uniform(2.0, 5.0), 6),
            "encrypt_time": round(random.uniform(1.0, 3.0), 6),
            "decrypt_time": round(random.uniform(1.0, 3.5), 6),
            "memory_usage": round(random.uniform(700.0, 1500.0), 2),
            "ciphertext_size": random.choice([5000, 8000, 12000]),
            "kem_level": random.choice(["ML-KEM-512", "ML-KEM-768", "ML-KEM-1024"]),
            "rsa_level": random.choice([2048, 4096]),
            "status": "error"
        }

        save_metrics(data, filepath)

    print(f"{count} entrées anormales générées")


# ============================================================
# VALIDATION CSV AVEC PANDAS
# ============================================================

def afficher_csv(filepath="metrics.csv"):
    """
    Personne B :
    Valide le fichier CSV avec pandas et affiche un aperçu.
    """

    if not os.path.exists(filepath):
        print("Fichier metrics.csv introuvable")
        return

    df = pd.read_csv(filepath)

    print("\nAperçu du fichier metrics.csv :")
    print(df.head(10).to_string(index=False))

    print("\nNombre total de sessions :", len(df))

    print("\nRépartition des statuts :")
    print(df["status"].value_counts())

    print("\nColonnes disponibles :")
    print(list(df.columns))


# ============================================================
# TEST MODULE 2
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("MODULE 2 — COLLECTE DES MÉTRIQUES")
    print("Salma A + Nabila B")
    print("=" * 60)

    message = "Message test pour métriques HPQS-AI"

    # Sessions normales avec les 3 niveaux ML-KEM
    for kem_level in [512, 768, 1024]:
        print(f"\nSession normale avec ML-KEM-{kem_level}")
        result = collect_normal_session(
            message=message,
            kem_level=kem_level,
            rsa_level=2048,
            filepath="metrics.csv"
        )
        print(result)

    # Sessions anormales simulées pour l'IA
    print("\nGénération des entrées anormales")
    generate_anomalous_entries(
        filepath="metrics.csv",
        count=5
    )

    # Validation finale du CSV
    afficher_csv("metrics.csv")

    print("\nMODULE 2 COMPLET : OK")