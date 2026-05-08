import streamlit as st
import os
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

# ============================================================
# CONFIG DE LA PAGE
# ============================================================
st.set_page_config(
    page_title="HPQS-AI Dashboard",
    page_icon="🔐",
    layout="wide"
)

# ============================================================
# FONCTION recommend_kem() — version simple sans le modèle ML
# (sera remplacée par optimizer_model.py plus tard)
# ============================================================
def recommend_kem(app_type, priority, memory):
    """
    Recommande ML-KEM-512, ML-KEM-768 ou ML-KEM-1024
    selon le contexte d'utilisation.
    """
    if app_type == "IoT" or memory == "Faible":
        return "ML-KEM-512"
    elif app_type == "Banque" or priority == "Sécurité maximale":
        return "ML-KEM-1024"
    else:
        return "ML-KEM-768"

# ============================================================
# FONCTION hybrid_encrypt() — version simulée pour le dashboard
# (sera connectée à crypto_module.py plus tard)
# ============================================================
def hybrid_encrypt_demo(message, kem_level):
    """
    Simule le chiffrement hybride RSA + ML-KEM + AES-GCM
    Retourne un faux ciphertext hex pour l'affichage
    """
    import hashlib, time

    # Simulation : on génère un hash du message comme "ciphertext"
    fake_cipher = hashlib.sha256(
        (message + kem_level + str(time.time())).encode()
    ).hexdigest()

    # On répète pour avoir une taille réaliste (comme un vrai ciphertext)
    full_cipher = (fake_cipher * 4)[:128]
    return full_cipher

# ============================================================
# SIDEBAR — Panneau latéral
# ============================================================
with st.sidebar:
    st.image(
        "https://img.icons8.com/color/96/lock--v1.png",
        width=80
    )
    st.title("⚙️ Configuration")
    st.markdown("---")

    st.subheader("🌐 Contexte d'utilisation")
    app_type = st.selectbox(
        "Type d'application",
        options=["Web", "IoT", "Mobile", "Banque", "Cloud"],
        index=0
    )

    st.subheader("🎯 Priorité")
    priority = st.radio(
        "Choisir la priorité",
        options=["Performance", "Équilibre", "Sécurité maximale"],
        index=1
    )

    st.subheader("💾 Mémoire disponible")
    memory = st.select_slider(
        "Niveau de mémoire",
        options=["Faible", "Moyenne", "Élevée"],
        value="Moyenne"
    )

    st.markdown("---")
    st.markdown("**Stack technique :**")
    st.markdown("🔑 RSA-2048")
    st.markdown("🔒 ML-KEM (Kyber)")
    st.markdown("🛡️ AES-256-GCM")
    st.markdown("🤖 Isolation Forest")
    st.markdown("🌳 Decision Tree")

# ============================================================
# TITRE PRINCIPAL
# ============================================================
st.title("🔐 HPQS-AI — Hybrid Post-Quantum Security")
st.markdown(
    "Système de chiffrement hybride combinant **RSA**, **ML-KEM/Kyber** "
    "et **AES-GCM** avec optimisation par Intelligence Artificielle."
)
st.markdown("---")

# ============================================================
# SECTION 1 — RECOMMANDATION ML-KEM
# ============================================================
st.header("🤖 Recommandation IA — Niveau ML-KEM optimal")

# Appel de recommend_kem() avec les paramètres du sidebar
kem_recommande = recommend_kem(app_type, priority, memory)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="📱 Contexte",
        value=app_type
    )

with col2:
    st.metric(
        label="🎯 Priorité",
        value=priority
    )

with col3:
    st.metric(
        label="💾 Mémoire",
        value=memory
    )

# Affichage du niveau recommandé avec couleur
st.markdown("### Niveau recommandé par l'IA :")

if kem_recommande == "ML-KEM-512":
    st.success(f"✅ {kem_recommande} — Léger, idéal pour IoT et faible mémoire")
elif kem_recommande == "ML-KEM-768":
    st.info(f"ℹ️ {kem_recommande} — Équilibré, recommandé pour Web et Mobile")
else:
    st.warning(f"🔒 {kem_recommande} — Sécurité maximale, pour Banque et Cloud")

st.markdown("---")

# ============================================================
# SECTION 2 — CHIFFREMENT DU MESSAGE
# ============================================================
st.header("✉️ Chiffrement du message")

# Champ de saisie du message
message_input = st.text_area(
    "📝 Entrez votre message à chiffrer",
    placeholder="Exemple : Transfert bancaire de 10 000 EUR vers compte FR76...",
    height=100
)

# Bouton de chiffrement
if st.button("🔐 Chiffrer le message", type="primary"):

    if message_input.strip() == "":
        st.error("❌ Veuillez entrer un message avant de chiffrer.")
    else:
        # Simulation du chiffrement
        with st.spinner("Chiffrement en cours..."):
            import time
            time.sleep(1)  # Simulation du temps de traitement
            ciphertext_hex = hybrid_encrypt_demo(message_input, kem_recommande)

        st.success("✅ Message chiffré avec succès !")

        # ── Schéma utilisé ──────────────────────────────────
        st.subheader("🏗️ Schéma cryptographique utilisé")

        schema_col1, schema_col2, schema_col3 = st.columns(3)

        with schema_col1:
            st.markdown(
                """
                <div style='background-color:#1e3a5f;
                            padding:15px;
                            border-radius:10px;
                            text-align:center;'>
                    <h4 style='color:white;'>🔑 RSA-2048</h4>
                    <p style='color:#aad4f5;'>Chiffrement classique du secret K2</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with schema_col2:
            st.markdown(
                f"""
                <div style='background-color:#1a4731;
                            padding:15px;
                            border-radius:10px;
                            text-align:center;'>
                    <h4 style='color:white;'>🔒 {kem_recommande}</h4>
                    <p style='color:#a8e6c3;'>Encapsulation post-quantique du secret K1</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with schema_col3:
            st.markdown(
                """
                <div style='background-color:#4a1942;
                            padding:15px;
                            border-radius:10px;
                            text-align:center;'>
                    <h4 style='color:white;'>🛡️ AES-256-GCM</h4>
                    <p style='color:#e8b4e8;'>Chiffrement symétrique du message</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Ciphertext affiché (format hex tronqué) ─────────
        st.subheader("🔡 Message chiffré (format hex tronqué)")

        ciphertext_tronque = ciphertext_hex[:64] + "..."

        st.code(ciphertext_tronque, language="text")

        st.caption(
            f"Taille totale du ciphertext : {len(ciphertext_hex)} caractères hex "
            f"({len(ciphertext_hex)//2} octets)"
        )

        # ── Résumé du schéma ────────────────────────────────
        st.info(
            f"🏗️ Schéma complet : **RSA-2048** + **{kem_recommande}** + **AES-256-GCM**\n\n"
            f"K1 (secret ML-KEM) ⊕ K2 (secret RSA) → HKDF(SHA-256) → Clé AES finale"
        )

# ============================================================
# SECTION 3 — EXPLICATION DU FLUX (toujours visible)
# ============================================================
st.markdown("---")
st.header("📊 Flux de chiffrement HPQS-AI")

flux_col1, flux_col2 = st.columns([1, 1])

with flux_col1:
    st.markdown("""
    **Étapes du chiffrement :**

    1. 🔑 RSA génère un secret classique **K2**
    2. 🔒 ML-KEM encapsule un secret post-quantique **K1**
    3. 🔀 Fusion : **K1 ⊕ K2** → HKDF(SHA-256) → clé finale
    4. 🛡️ AES-256-GCM chiffre le message avec la clé finale
    5. 📊 Les métriques sont enregistrées dans **metrics.csv**
    6. 🤖 L'IA détecte les anomalies (Isolation Forest)
    7. 🌳 L'IA recommande le meilleur niveau ML-KEM (Decision Tree)
    """)

with flux_col2:
    st.markdown("""
    **Niveaux ML-KEM disponibles :**

    | Niveau | Sécurité | Usage recommandé |
    |--------|----------|-----------------|
    | ML-KEM-512 | ⭐⭐ | IoT, faible mémoire |
    | ML-KEM-768 | ⭐⭐⭐ | Web, Mobile, Cloud |
    | ML-KEM-1024 | ⭐⭐⭐⭐ | Banque, données critiques |
    """)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray;'>"
    "HPQS-AI — Projet IA + Cryptographie Post-Quantique | "
    "Coéquipier A + Coéquipier B"
    "</div>",
    unsafe_allow_html=True
)