# ============================================================
# metrics_collector.py — Module 2
# Projet HPQS-AI — Collecte des métriques de performance
# ============================================================

import time
import csv
import os
import pandas as pd
from crypto_module import generate_rsa_keys, hybrid_encrypt


# ─────────────────────────────────────────
# MESURE 1 — keygen_time
# ─────────────────────────────────────────
def measure_keygen_time():
    start = time.time()
    rsa_private, rsa_public = generate_rsa_keys()
    end = time.time()
    keygen_time = round(end - start, 6)
    return keygen_time, rsa_private, rsa_public


# ─────────────────────────────────────────
# MESURE 2 — encrypt_time + ciphertext_size
# ─────────────────────────────────────────
def measure_encrypt_time(message, rsa_public, K1):
    start = time.time()
    bundle = hybrid_encrypt(message, rsa_public, K1)
    end = time.time()
    encrypt_time = round(end - start, 6)
    ciphertext_size = len(bundle["ciphertext"])
    return encrypt_time, ciphertext_size, bundle


# ─────────────────────────────────────────
# FONCTION PRINCIPALE — save_metrics()
# ─────────────────────────────────────────
def save_metrics(data, filepath="metrics.csv"):
    file_exists = os.path.exists(filepath)

    with open(filepath, mode='a', newline='') as csvfile:
        fieldnames = [
            "keygen_time",
            "encrypt_time",
            "ciphertext_size",
            "kem_level"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(data)

    print(f"✅ Métriques sauvegardées dans {filepath}")


# ─────────────────────────────────────────
# VALIDATION — Afficher le CSV avec pandas
# ─────────────────────────────────────────
def afficher_csv(filepath="metrics.csv"):
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        print("\n📊 Aperçu du fichier metrics.csv :")
        print(df.to_string())
    else:
        print("❌ Fichier metrics.csv introuvable")


# ─────────────────────────────────────────
# TEST
# ─────────────────────────────────────────
if __name__ == "__main__":

    K1_simule = os.urandom(32)

    keygen_time, rsa_private, rsa_public = measure_keygen_time()
    print(f"✅ keygen_time     : {keygen_time} s")

    message = "Message test pour métriques"
    encrypt_time, ciphertext_size, bundle = measure_encrypt_time(
        message, rsa_public, K1_simule
    )
    print(f"✅ encrypt_time    : {encrypt_time} s")
    print(f"✅ ciphertext_size : {ciphertext_size} octets")

    # kem_level : 512 (IoT/léger) | 768 (web/équilibré) | 1024 (banque/max)
    # sera mis à jour dynamiquement quand binôme intègre ML-KEM
    data = {
        "keygen_time"    : keygen_time,
        "encrypt_time"   : encrypt_time,
        "ciphertext_size": ciphertext_size,
        "kem_level"      : "ML-KEM-512"
    }

    save_metrics(data)
    afficher_csv()