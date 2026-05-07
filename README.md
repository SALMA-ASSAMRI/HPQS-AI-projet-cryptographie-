# HPQS-AI — Hybrid Post-Quantum Security System

## Description du projet

HPQS-AI est un système de sécurité hybride post-quantique qui combine
la cryptographie classique (RSA) et post-quantique (ML-KEM/Kyber)
avec AES-256-GCM pour chiffrer des messages de façon sécurisée.
Le système intègre également une couche intelligence artificielle
pour détecter les anomalies et optimiser automatiquement
le niveau de sécurité selon le contexte d'utilisation.

---

## Équipe

- **Coéquipier A** — SALMA ASSAMRI
- **Coéquipier B** — Nabila ABOUILAAZ

---

## Modules implémentés

### Module 1 — Cryptographie Hybride (`crypto_module.py`)

Ce module constitue le cœur cryptographique du projet.
Il implémente un système de chiffrement hybride en trois couches :
RSA-2048 pour le chiffrement du secret classique,
ML-KEM/Kyber pour le secret post-quantique,
et AES-256-GCM pour le chiffrement final du message.

#### Fonctionnement
K2 (aléatoire) ──► RSA-OAEP ──► K2_encrypted
K1 (ML-KEM)   ──┐
K2             ──┤ XOR ──► HKDF ──► clé AES ──► AES-256-GCM ──► message chiffré

#### Fonctions principales
| Fonction | Description |
|---|---|
| `generate_rsa_keys()` | Génère une paire de clés RSA-2048 |
| `generate_K2()` | Génère un secret aléatoire de 32 octets |
| `encrypt_K2_with_RSA()` | Chiffre K2 avec RSA-OAEP SHA-256 |
| `fuse_secrets_HKDF()` | Fusionne K1 XOR K2 via HKDF SHA-256 |
| `hybrid_encrypt()` | Chiffre un message complet (fonction principale) |
<<<<<<< HEAD

#### Exécution
=======
Exécution :

>>>>>>> fc4e3a561acc63a8570a0b9658e545346330d94d
```bash
python crypto_module.py
```
### Module 2 — Collecte des Métriques (`metrics_collector.py`)
Ce module mesure et enregistre les performances de chaque session
cryptographique dans un fichier `metrics.csv`.
Ces données seront utilisées par les modules IA (détection d'anomalies
et optimisation ML-KEM).
#### Fonctions principales
| Fonction | Description |
|---|---|
| `measure_keygen_time()` | Mesure le temps de génération des clés RSA |
| `measure_encrypt_time()` | Mesure le temps de chiffrement + taille du chiffré |
| `save_metrics()` | Sauvegarde les métriques dans metrics.csv |
| `afficher_csv()` | Affiche les données enregistrées avec pandas |
#### Métriques enregistrées dans `metrics.csv`
| Champ | Description | Exemple |
|---|---|---|
| `keygen_time` | Temps génération clés RSA (en secondes) | 0.264741 |
| `encrypt_time` | Temps chiffrement message (en secondes) | 0.037485 |
| `ciphertext_size` | Taille du message chiffré (en octets) | 44 |
| `kem_level` | Niveau ML-KEM utilisé | ML-KEM-512 |
#### Exécution
```bash
python metrics_collector.py
```

---
## Installation
```bash
pip install cryptography pandas psutil scikit-learn joblib matplotlib streamlit
```

---
## Structure du projet
HPQS-AI-projet-cryptographie/
├── crypto_module.py        # Module 1 — Cryptographie hybride
├── metrics_collector.py    # Module 2 — Collecte des métriques
├── anomaly_detection.py    # Module 3 — Détection d'anomalies (IA)
├── optimizer_model.py      # Module 4 — Optimisation ML-KEM (IA)
├── app.py                  # Module 5 — Dashboard Streamlit
├── metrics.csv             # Données des sessions générées
├── data/                   # Dossier des données
└── requirements.txt        # Bibliothèques requises
## Stack Technique
| Outil | Rôle |
|---|---|
| `cryptography` | RSA + AES-GCM + HKDF |
| `pandas` | Lecture et validation CSV |
| `scikit-learn` | Isolation Forest + Decision Tree |
| `streamlit` | Dashboard interactif |
| `psutil` | Mesure mémoire système |
| `joblib` | Sauvegarde des modèles IA |


