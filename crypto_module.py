## ============================================================
# crypto_module.py — Module 1
# Projet HPQS-AI — Cryptographie Hybride RSA + ML-KEM + AES
# ============================================================

import os
import oqs

from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


# ============================================================
# PARTIE A 
# ============================================================

# ─────────────────────────────────────────
# ÉTAPE 1 — Générer la paire de clés RSA
# ─────────────────────────────────────────
def generate_rsa_keys():
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048
    )
    public_key = private_key.public_key()
    return private_key, public_key


# ─────────────────────────────────────────
# ÉTAPE 2 — Générer le secret K2
# ─────────────────────────────────────────
def generate_K2():
    return os.urandom(32)


# ─────────────────────────────────────────
# ÉTAPE 3 — Chiffrer K2 avec RSA-OAEP
# ─────────────────────────────────────────
def encrypt_K2_with_RSA(K2, rsa_public_key):
    K2_encrypted = rsa_public_key.encrypt(
        K2,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    return K2_encrypted


# ─────────────────────────────────────────
# ÉTAPE 4 — Fusion HKDF(K1 XOR K2)
# ─────────────────────────────────────────
def fuse_secrets_HKDF(K1, K2):
    """
    Fusionne K1 et K2 avec XOR puis HKDF.
    Le résultat est une clé AES-256 de 32 octets.
    """

    K1 = K1[:32]
    K2 = K2[:32]

    K_combined = bytes(a ^ b for a, b in zip(K1, K2))

    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=b"HPQS-AI hybrid key"
    )

    return hkdf.derive(K_combined)


# ─────────────────────────────────────────
# ÉTAPE 5 — hybrid_encrypt()
# ─────────────────────────────────────────
def hybrid_encrypt(message, rsa_public_key, K1):
    """
    Fonction de chiffrement hybride côté Salma A.

    Entrées :
    - message : texte à chiffrer
    - rsa_public_key : clé publique RSA
    - K1 : secret post-quantique généré par ML-KEM côté Personne B

    Sortie :
    - bundle contenant K2 chiffré, ciphertext AES-GCM et nonce
    """

    # 1. Générer K2
    K2 = generate_K2()

    # 2. Chiffrer K2 avec RSA
    K2_encrypted = encrypt_K2_with_RSA(K2, rsa_public_key)

    # 3. Fusionner K1 + K2 → clé AES
    aes_key = fuse_secrets_HKDF(K1, K2)

    # 4. Chiffrer le message avec AES-256-GCM
    nonce = os.urandom(12)
    aesgcm = AESGCM(aes_key)
    ciphertext = aesgcm.encrypt(nonce, message.encode("utf-8"), None)

    # 5. Retourner le bundle
    return {
        "K2_encrypted": K2_encrypted,
        "ciphertext": ciphertext,
        "nonce": nonce
    }


# ============================================================
# PARTIE B : ML-KEM / liboqs
# ============================================================

def get_kem_name(kem_level=512):
    """
    Convertit 512, 768, 1024 vers ML-KEM-512, ML-KEM-768, ML-KEM-1024.
    """

    if isinstance(kem_level, int):
        return f"ML-KEM-{kem_level}"

    if isinstance(kem_level, str):
        if kem_level.startswith("ML-KEM-"):
            return kem_level
        if kem_level in ["512", "768", "1024"]:
            return f"ML-KEM-{kem_level}"

    raise ValueError("kem_level doit être 512, 768, 1024 ou ML-KEM-XXX")


def generate_mlkem_keys(kem_level=512):
    """
    Personne B :
    Génère une paire de clés ML-KEM avec liboqs.
    """

    kem_name = get_kem_name(kem_level)

    with oqs.KeyEncapsulation(kem_name) as kem:
        mlkem_public_key = kem.generate_keypair()
        mlkem_secret_key = kem.export_secret_key()

    return mlkem_public_key, mlkem_secret_key


def encapsulate_K1(mlkem_public_key, kem_level=512):
    """
    Personne B :
    Encapsule le secret partagé K1 avec la clé publique ML-KEM.

    Retourne :
    - ciphertext_kem : chiffré ML-KEM
    - K1 : secret partagé côté chiffrement
    """

    kem_name = get_kem_name(kem_level)

    with oqs.KeyEncapsulation(kem_name) as kem:
        ciphertext_kem, K1 = kem.encap_secret(mlkem_public_key)

    return ciphertext_kem, K1


def decapsulate_K1(ciphertext_kem, mlkem_secret_key, kem_level=512):
    """
    Personne B :
    Décapsule K1 avec la clé privée ML-KEM.
    """

    kem_name = get_kem_name(kem_level)

    with oqs.KeyEncapsulation(kem_name, mlkem_secret_key) as kem:
        K1 = kem.decap_secret(ciphertext_kem)

    return K1


# ============================================================
# PARTIE B — Déchiffrement RSA de K2
# ============================================================

def decrypt_K2_with_RSA(K2_encrypted, rsa_private_key):
    """
    Personne B :
    Déchiffre K2 avec la clé privée RSA.
    """

    K2 = rsa_private_key.decrypt(
        K2_encrypted,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )

    return K2


# ============================================================
# PARTIE B — AES-256-GCM Déchiffrement
# ============================================================

def decrypt_message_AES_GCM(ciphertext, nonce, aes_key):
    """
    Personne B :
    Déchiffre le message avec AES-256-GCM.
    """

    aesgcm = AESGCM(aes_key)
    plaintext = aesgcm.decrypt(nonce, ciphertext, None)

    return plaintext.decode("utf-8")


# ============================================================
# PARTIE B — hybrid_decrypt()
# ============================================================

def hybrid_decrypt(bundle, rsa_private_key, mlkem_secret_key, kem_level=512):
    """
    Personne B :
    Déchiffrement hybride complet.

    Étapes :
    1. Décapsuler K1 avec ML-KEM
    2. Déchiffrer K2 avec RSA-OAEP
    3. Fusionner K1 et K2 avec HKDF
    4. Déchiffrer le message avec AES-256-GCM
    """

    # 1. Récupérer le ciphertext ML-KEM
    ciphertext_kem = bundle["ciphertext_kem"]

    # 2. Décapsuler K1
    K1 = decapsulate_K1(
        ciphertext_kem,
        mlkem_secret_key,
        kem_level
    )

    # 3. Déchiffrer K2
    K2 = decrypt_K2_with_RSA(
        bundle["K2_encrypted"],
        rsa_private_key
    )

    # 4. Fusionner K1 + K2 pour retrouver la clé AES
    aes_key = fuse_secrets_HKDF(K1, K2)

    # 5. Déchiffrer le message
    message_decrypted = decrypt_message_AES_GCM(
        bundle["ciphertext"],
        bundle["nonce"],
        aes_key
    )

    return message_decrypted


# ============================================================
# FONCTION OPTIONNELLE — TEST COMPLET A + B
# ============================================================

def hybrid_encrypt_full(message, kem_level=512):
    """
    Fonction pratique pour tester tout le cycle :
    RSA + ML-KEM + HKDF + AES-GCM.

    Elle combine automatiquement le travail de Salma A et Personne B.
    """

    # 1. Générer RSA
    rsa_private_key, rsa_public_key = generate_rsa_keys()

    # 2. Générer ML-KEM
    mlkem_public_key, mlkem_secret_key = generate_mlkem_keys(kem_level)

    # 3. Encapsuler K1 avec ML-KEM
    ciphertext_kem, K1 = encapsulate_K1(
        mlkem_public_key,
        kem_level
    )

    # 4. Chiffrer le message avec la fonction de Salma
    bundle = hybrid_encrypt(
        message,
        rsa_public_key,
        K1
    )

    # 5. Ajouter ciphertext_kem dans le bundle pour le déchiffrement
    bundle["ciphertext_kem"] = ciphertext_kem
    bundle["kem_level"] = kem_level

    return bundle, rsa_private_key, rsa_public_key, mlkem_secret_key, mlkem_public_key


# ============================================================
# TEST — Lance avec : python crypto_module.py
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("TEST MODULE 1 — INTÉGRATION  A + B")
    print("=" * 60)

    message = "Bonjour, ceci est un message secret HPQS-AI !"

    for kem_level in [512, 768, 1024]:

        print("\n" + "-" * 60)
        print(f"TEST AVEC ML-KEM-{kem_level}")
        print("-" * 60)

        # Chiffrement complet
        bundle, rsa_private_key, rsa_public_key, mlkem_secret_key, mlkem_public_key = hybrid_encrypt_full(
            message,
            kem_level
        )

        print("Chiffrement réussi")
        print(f"Taille ciphertext AES     : {len(bundle['ciphertext'])} octets")
        print(f"Taille ciphertext ML-KEM  : {len(bundle['ciphertext_kem'])} octets")
        print(f"K2 encrypted début        : {bundle['K2_encrypted'][:10].hex()}...")
        print(f"Nonce                     : {bundle['nonce'].hex()}")

        # Déchiffrement complet
        decrypted_message = hybrid_decrypt(
            bundle,
            rsa_private_key,
            mlkem_secret_key,
            kem_level
        )

        print("Message original  :", message)
        print("Message déchiffré :", decrypted_message)

        if message == decrypted_message:
            print("TEST RÉUSSI : chiffrement → déchiffrement fonctionne")
        else:
            print("TEST ÉCHOUÉ : les messages sont différents")

    print("\n" + "=" * 60)
    print("MODULE 1 COMPLET : OK")
    print("=" * 60)