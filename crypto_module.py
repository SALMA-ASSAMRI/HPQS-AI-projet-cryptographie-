# ============================================================
# crypto_module.py — Module 1 
# Projet HPQS-AI — Cryptographie Hybride RSA + Kyber + AES
# ============================================================

import os
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


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
    # XOR les deux secrets
    K_combined = bytes(a ^ b for a, b in zip(K1, K2))
    # HKDF pour dériver la clé AES finale
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
    # 1. Générer K2
    K2 = generate_K2()
    # 2. Chiffrer K2 avec RSA
    K2_encrypted = encrypt_K2_with_RSA(K2, rsa_public_key)
    # 3. Fusionner K1 + K2 → clé AES
    aes_key = fuse_secrets_HKDF(K1, K2)
    # 4. Chiffrer le message avec AES-256-GCM
    nonce = os.urandom(12)
    aesgcm = AESGCM(aes_key)
    ciphertext = aesgcm.encrypt(nonce, message.encode('utf-8'), None)
    # 5. Retourner le bundle
    return {
        "K2_encrypted" : K2_encrypted,
        "ciphertext"   : ciphertext,
        "nonce"        : nonce
    }


# ─────────────────────────────────────────
# TEST — Lance avec : python crypto_module.py
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  TEST MODULE 1 — Coéquipier A")
    print("=" * 50)

    # Générer clés RSA
    rsa_private, rsa_public = generate_rsa_keys()
    print("✅ Clés RSA-2048 générées")

    # Simuler K1 (normalement vient de Kyber/binôme)
    K1_simule = os.urandom(32)
    print("✅ K1 simulé (32 octets)")

    # Chiffrer un message
    message = "Bonjour, ceci est un message secret !"
    bundle = hybrid_encrypt(message, rsa_public, K1_simule)

    print(f"✅ Message chiffré avec succès !")
    print(f"   Taille ciphertext  : {len(bundle['ciphertext'])} octets")
    print(f"   K2_encrypted (début): {bundle['K2_encrypted'][:10].hex()}...")
    print(f"   Nonce              : {bundle['nonce'].hex()}")
    print("=" * 50)
    print("  MODULE 1 COÉQUIPIER A : OK ✅")
    print("=" * 50)