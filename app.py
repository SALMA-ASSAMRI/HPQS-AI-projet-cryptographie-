# ============================================================
# app.py — Module 5
# Projet HPQS-AI — Dashboard Streamlit propre et lisible
# Salma A + Nabila B
# Chiffrement + Déchiffrement avec deux boutons séparés
# ============================================================

import os
import time
import pandas as pd
import streamlit as st

from optimizer_model import recommend_kem
from crypto_module import hybrid_encrypt_full, hybrid_decrypt


# ============================================================
# CONFIGURATION PAGE
# ============================================================

st.set_page_config(
    page_title="HPQS-AI Dashboard",
    page_icon="H",
    layout="wide"
)


# ============================================================
# STYLE CSS
# ============================================================

st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(135deg, #eef2ff 0%, #f8fafc 45%, #e0f2fe 100%);
            background-size: 200% 200%;
            animation: bgMove 16s ease infinite;
            color: #0f172a;
        }

        @keyframes bgMove {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        @keyframes fadeUp {
            from {
                opacity: 0;
                transform: translateY(14px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .hero {
            background: linear-gradient(135deg, #0f172a, #1e293b);
            border-radius: 24px;
            padding: 36px 40px;
            margin-bottom: 28px;
            box-shadow: 0 22px 55px rgba(15, 23, 42, 0.22);
            animation: fadeUp 0.7s ease;
        }

        .hero-title {
            color: white;
            font-size: 40px;
            font-weight: 850;
            letter-spacing: -1px;
            margin-bottom: 10px;
        }

        .hero-subtitle {
            color: #cbd5e1;
            font-size: 16px;
            line-height: 1.6;
            max-width: 850px;
        }

        .section-title {
            color: #0f172a;
            font-size: 22px;
            font-weight: 850;
            margin-top: 24px;
            margin-bottom: 6px;
        }

        .section-subtitle {
            color: #475569;
            font-size: 14px;
            margin-bottom: 16px;
            line-height: 1.5;
        }

        .mini-card {
            background: #ffffff;
            border: 1px solid #dbe3ef;
            border-radius: 18px;
            padding: 20px;
            min-height: 112px;
            box-shadow: 0 10px 26px rgba(15, 23, 42, 0.07);
            transition: all 0.25s ease;
            animation: fadeUp 0.6s ease;
        }

        .mini-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 18px 40px rgba(15, 23, 42, 0.13);
        }

        .mini-label {
            color: #64748b;
            font-size: 13px;
            font-weight: 700;
            margin-bottom: 10px;
        }

        .mini-value {
            color: #0f172a;
            font-size: 28px;
            font-weight: 850;
            letter-spacing: -0.5px;
        }

        .kem-box {
            padding: 26px;
            border-radius: 20px;
            text-align: center;
            font-size: 31px;
            font-weight: 850;
            letter-spacing: -0.5px;
            animation: pulseSoft 2.8s ease-in-out infinite;
            margin-bottom: 20px;
        }

        @keyframes pulseSoft {
            0% { transform: scale(1); }
            50% { transform: scale(1.012); }
            100% { transform: scale(1); }
        }

        .kem-desc {
            font-size: 14px;
            font-weight: 600;
            margin-top: 10px;
            line-height: 1.5;
        }

        .kem-512 {
            background: #ecfdf5;
            color: #047857;
            border: 1px solid #86efac;
        }

        .kem-768 {
            background: #eff6ff;
            color: #1d4ed8;
            border: 1px solid #93c5fd;
        }

        .kem-1024 {
            background: #fff7ed;
            color: #c2410c;
            border: 1px solid #fdba74;
        }

        .status-success {
            background: #ecfdf5;
            color: #047857;
            border: 1px solid #86efac;
            border-radius: 16px;
            padding: 15px 18px;
            font-weight: 800;
            margin: 14px 0;
        }

        .status-error {
            background: #fef2f2;
            color: #b91c1c;
            border: 1px solid #fca5a5;
            border-radius: 16px;
            padding: 15px 18px;
            font-weight: 800;
            margin: 14px 0;
        }

        .crypto-box {
            background: #ffffff;
            border: 1px solid #dbe3ef;
            border-radius: 18px;
            padding: 18px;
            margin-top: 14px;
            margin-bottom: 18px;
            box-shadow: 0 10px 26px rgba(15, 23, 42, 0.06);
        }

        .crypto-label {
            color: #64748b;
            font-size: 13px;
            font-weight: 700;
            margin-bottom: 8px;
        }

        .crypto-value {
            color: #0f172a;
            font-size: 16px;
            font-weight: 700;
            line-height: 1.6;
        }

        div.stButton > button {
            background: linear-gradient(135deg, #0f172a, #334155);
            color: white !important;
            border: none;
            border-radius: 14px;
            height: 46px;
            font-weight: 800;
            transition: all 0.25s ease;
        }

        div.stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 14px 26px rgba(15, 23, 42, 0.24);
        }

        div[data-testid="stTextArea"] textarea {
            background-color: #ffffff !important;
            color: #0f172a !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 16px !important;
            font-weight: 600 !important;
            caret-color: #0f172a !important;
        }

        div[data-testid="stTextArea"] textarea::placeholder {
            color: #64748b !important;
            opacity: 1 !important;
        }

        div[data-testid="stTextArea"] label {
            color: #0f172a !important;
            font-weight: 700 !important;
        }

        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #dbe3ef;
            border-radius: 18px;
            padding: 18px;
            box-shadow: 0 10px 26px rgba(15, 23, 42, 0.07);
        }

        div[data-testid="stMetricLabel"] p {
            color: #475569 !important;
            font-weight: 700 !important;
        }

        div[data-testid="stMetricValue"] {
            color: #0f172a !important;
            font-weight: 850 !important;
        }

        section[data-testid="stSidebar"] {
            background: #0f172a;
        }

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] span {
            color: #e5e7eb !important;
        }

        hr {
            border-color: rgba(15, 23, 42, 0.12);
            margin-top: 24px;
            margin-bottom: 24px;
        }

        .footer {
            text-align: center;
            color: #64748b;
            font-size: 13px;
            margin-top: 30px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

def get_kem_number(kem_text):
    return int(str(kem_text).replace("ML-KEM-", ""))


def kem_description(kem_level):
    if kem_level == "ML-KEM-512":
        return "Léger et rapide. Adapté aux environnements IoT et aux systèmes avec peu de mémoire."
    if kem_level == "ML-KEM-768":
        return "Bon équilibre entre performance et sécurité. Adapté aux usages Web, Mobile et Cloud standard."
    if kem_level == "ML-KEM-1024":
        return "Sécurité maximale. Recommandé pour Banque, Cloud critique et données sensibles."
    return "Niveau recommandé par le modèle IA."


def kem_css_class(kem_level):
    if kem_level == "ML-KEM-512":
        return "kem-512"
    if kem_level == "ML-KEM-768":
        return "kem-768"
    if kem_level == "ML-KEM-1024":
        return "kem-1024"
    return ""


def charger_sessions():
    if os.path.exists("metrics_with_anomalies.csv"):
        return pd.read_csv("metrics_with_anomalies.csv")
    if os.path.exists("metrics.csv"):
        return pd.read_csv("metrics.csv")
    return pd.DataFrame()


def sauvegarder_session_dashboard(data, filepath="dashboard_sessions.csv"):
    df_new = pd.DataFrame([data])

    if os.path.exists(filepath):
        df_old = pd.read_csv(filepath)
        df_final = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_final = df_new

    df_final.to_csv(filepath, index=False)


def resultat_ia_simple(encrypt_time, decrypt_time, memory_usage, ciphertext_size):
    if encrypt_time > 1 or decrypt_time > 1 or memory_usage > 700 or ciphertext_size > 5000:
        return "Anomalie"
    return "Normal"


def html_metric(label, value):
    st.markdown(
        f"""
        <div class="mini-card">
            <div class="mini-label">{label}</div>
            <div class="mini-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def init_crypto_state():
    default_values = {
        "crypto_bundle": None,
        "rsa_private_key": None,
        "mlkem_secret_key": None,
        "kem_number": None,
        "original_message": None,
        "selected_kem": None,
        "encrypt_time": None,
        "decrypt_time": None,
        "memory_usage": None,
        "ciphertext_size": None,
        "message_encrypted": False,
        "message_decrypted": False,
        "decrypted_message": None,
        "session_saved": False
    }

    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_crypto_state():
    keys_to_reset = [
        "crypto_bundle",
        "rsa_private_key",
        "mlkem_secret_key",
        "kem_number",
        "original_message",
        "selected_kem",
        "encrypt_time",
        "decrypt_time",
        "memory_usage",
        "ciphertext_size",
        "decrypted_message"
    ]

    for key in keys_to_reset:
        st.session_state[key] = None

    st.session_state.message_encrypted = False
    st.session_state.message_decrypted = False
    st.session_state.session_saved = False


def afficher_sessions_precedentes():
    df = charger_sessions()

    st.markdown('<div class="section-title">Sessions précédentes</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Historique des sessions générées par les modules de métriques et de détection IA.</div>',
        unsafe_allow_html=True
    )

    if df.empty:
        st.info("Aucune session précédente trouvée.")
        return

    colonnes_preferees = [
        "keygen_time",
        "encrypt_time",
        "decrypt_time",
        "memory_usage",
        "ciphertext_size",
        "kem_level",
        "rsa_level",
        "status",
        "anomaly_prediction",
        "anomaly_score",
        "ai_result"
    ]

    colonnes_existantes = [col for col in colonnes_preferees if col in df.columns]
    st.dataframe(df[colonnes_existantes], use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Répartition IA</div>', unsafe_allow_html=True)
        if "ai_result" in df.columns:
            st.bar_chart(df["ai_result"].value_counts())
        elif "status" in df.columns:
            st.bar_chart(df["status"].value_counts())
        else:
            st.info("Aucune donnée IA disponible.")

    with col2:
        st.markdown('<div class="section-title">Répartition ML-KEM</div>', unsafe_allow_html=True)
        if "kem_level" in df.columns:
            st.bar_chart(df["kem_level"].value_counts())
        else:
            st.info("Aucune donnée ML-KEM disponible.")


# ============================================================
# INITIALISATION SESSION
# ============================================================

init_crypto_state()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("HPQS-AI")
st.sidebar.markdown("Configuration du contexte")

app_type = st.sidebar.selectbox(
    "Type d'application",
    ["IoT", "Web", "Mobile", "Cloud", "Banque"]
)

priority = st.sidebar.selectbox(
    "Priorité",
    ["Performance", "Équilibre", "Sécurité maximale"]
)

memory_label = st.sidebar.selectbox(
    "Mémoire disponible",
    ["Faible", "Moyenne", "Élevée"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("Technologies")
st.sidebar.write("RSA-2048")
st.sidebar.write("ML-KEM")
st.sidebar.write("AES-256-GCM")
st.sidebar.write("Isolation Forest")
st.sidebar.write("Decision Tree")


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">HPQS-AI Dashboard</div>
        <div class="hero-subtitle">
            Système hybride post-quantique combinant RSA, ML-KEM et AES-GCM,
            avec intelligence artificielle pour recommander le niveau de sécurité
            et détecter les anomalies.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# RECOMMANDATION IA
# ============================================================

recommended_kem = recommend_kem(app_type, priority, memory_label)

col1, col2, col3 = st.columns(3)

with col1:
    html_metric("Contexte", app_type)

with col2:
    html_metric("Priorité", priority)

with col3:
    html_metric("Mémoire", memory_label)

st.markdown('<div class="section-title">Recommandation IA</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Le modèle choisit automatiquement le niveau ML-KEM le plus adapté au contexte.</div>',
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <div class="kem-box {kem_css_class(recommended_kem)}">
        {recommended_kem}
        <div class="kem-desc">{kem_description(recommended_kem)}</div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")


# ============================================================
# CHIFFREMENT + DÉCHIFFREMENT
# ============================================================

st.markdown('<div class="section-title">Chiffrement et déchiffrement du message</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Entrez un message, chiffrez-le, puis utilisez le bouton de déchiffrement pour retrouver le message original.</div>',
    unsafe_allow_html=True
)

message = st.text_area(
    "Message à chiffrer",
    placeholder="Exemple : Bonjour, ceci est un test HPQS-AI"
)

col_encrypt, col_reset = st.columns([1, 4])

with col_encrypt:
    chiffrer = st.button("Chiffrer le message", type="primary")

with col_reset:
    reset = st.button("Réinitialiser")

if reset:
    reset_crypto_state()
    st.rerun()


if chiffrer:

    if not message.strip():
        st.markdown(
            '<div class="status-error">Veuillez entrer un message avant de chiffrer.</div>',
            unsafe_allow_html=True
        )

    else:
        kem_number = get_kem_number(recommended_kem)

        start_encrypt = time.time()

        bundle, rsa_private_key, rsa_public_key, mlkem_secret_key, mlkem_public_key = hybrid_encrypt_full(
            message,
            kem_number
        )

        end_encrypt = time.time()

        encrypt_time = round(end_encrypt - start_encrypt, 6)
        memory_usage = 112.0
        ciphertext_size = len(bundle["ciphertext"])

        st.session_state.crypto_bundle = bundle
        st.session_state.rsa_private_key = rsa_private_key
        st.session_state.mlkem_secret_key = mlkem_secret_key
        st.session_state.kem_number = kem_number
        st.session_state.original_message = message
        st.session_state.selected_kem = recommended_kem
        st.session_state.encrypt_time = encrypt_time
        st.session_state.decrypt_time = None
        st.session_state.memory_usage = memory_usage
        st.session_state.ciphertext_size = ciphertext_size
        st.session_state.message_encrypted = True
        st.session_state.message_decrypted = False
        st.session_state.decrypted_message = None
        st.session_state.session_saved = False

        st.markdown(
            '<div class="status-success">Message chiffré avec succès.</div>',
            unsafe_allow_html=True
        )


if st.session_state.message_encrypted:

    st.markdown('<div class="section-title">Schéma cryptographique</div>', unsafe_allow_html=True)

    schema1, schema2, schema3 = st.columns(3)

    with schema1:
        st.info("RSA-2048\n\nProtection du secret classique K2")

    with schema2:
        st.info(f"{st.session_state.selected_kem}\n\nEncapsulation du secret post-quantique K1")

    with schema3:
        st.info("AES-256-GCM\n\nChiffrement symétrique du message")

    st.markdown('<div class="section-title">Message chiffré</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Le message est affiché sous forme hexadécimale tronquée pour rester lisible.</div>',
        unsafe_allow_html=True
    )

    ciphertext_hex = st.session_state.crypto_bundle["ciphertext"].hex()
    ciphertext_display = ciphertext_hex[:90] + "..." if len(ciphertext_hex) > 90 else ciphertext_hex

    st.code(ciphertext_display)

    dechiffrer = st.button("Déchiffrer le message")

    if dechiffrer:
        start_decrypt = time.time()

        decrypted_message = hybrid_decrypt(
            st.session_state.crypto_bundle,
            st.session_state.rsa_private_key,
            st.session_state.mlkem_secret_key,
            st.session_state.kem_number
        )

        end_decrypt = time.time()

        decrypt_time = round(end_decrypt - start_decrypt, 6)

        st.session_state.decrypted_message = decrypted_message
        st.session_state.decrypt_time = decrypt_time
        st.session_state.message_decrypted = True


if st.session_state.message_decrypted:

    st.markdown(
        '<div class="status-success">Message déchiffré avec succès.</div>',
        unsafe_allow_html=True
    )

    ai_result = resultat_ia_simple(
        st.session_state.encrypt_time,
        st.session_state.decrypt_time,
        st.session_state.memory_usage,
        st.session_state.ciphertext_size
    )

    st.markdown('<div class="section-title">Message déchiffré</div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="crypto-box">
            <div class="crypto-label">Résultat du déchiffrement</div>
            <div class="crypto-value">{st.session_state.decrypted_message}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.session_state.decrypted_message == st.session_state.original_message:
        st.success("Le message déchiffré correspond au message original.")
    else:
        st.error("Le message déchiffré ne correspond pas au message original.")

    st.markdown('<div class="section-title">Métriques de la session</div>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric("Chiffrement", f"{st.session_state.encrypt_time} s")

    with m2:
        st.metric("Déchiffrement", f"{st.session_state.decrypt_time} s")

    with m3:
        st.metric("Mémoire", f"{st.session_state.memory_usage} MB")

    with m4:
        st.metric("Ciphertext", f"{st.session_state.ciphertext_size} octets")

    st.markdown('<div class="section-title">Résultat IA</div>', unsafe_allow_html=True)

    if ai_result == "Normal":
        st.markdown('<div class="status-success">Résultat : Normal</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-error">Résultat : Anomalie détectée</div>', unsafe_allow_html=True)

    if not st.session_state.session_saved:
        session_data = {
            "app_type": app_type,
            "priority": priority,
            "memory": memory_label,
            "kem_level": st.session_state.selected_kem,
            "encrypt_time": st.session_state.encrypt_time,
            "decrypt_time": st.session_state.decrypt_time,
            "memory_usage": st.session_state.memory_usage,
            "ciphertext_size": st.session_state.ciphertext_size,
            "ai_result": ai_result
        }

        sauvegarder_session_dashboard(session_data)
        st.session_state.session_saved = True


# ============================================================
# FLUX DU SYSTÈME
# ============================================================

st.markdown("---")
st.markdown('<div class="section-title">Flux du système HPQS-AI</div>', unsafe_allow_html=True)

st.markdown(
    """
1. RSA-2048 protège le secret classique K2.  
2. ML-KEM encapsule le secret post-quantique K1.  
3. K1 et K2 sont fusionnés avec XOR puis HKDF.  
4. AES-256-GCM chiffre le message final.  
5. Le bouton de déchiffrement utilise les clés conservées temporairement dans la session.  
6. Les métriques sont enregistrées.  
7. Isolation Forest détecte les anomalies.  
8. Decision Tree recommande le niveau ML-KEM optimal.
"""
)


# ============================================================
# TABLEAU ML-KEM
# ============================================================

st.markdown('<div class="section-title">Niveaux ML-KEM</div>', unsafe_allow_html=True)

df_kem = pd.DataFrame([
    {
        "Niveau": "ML-KEM-512",
        "Utilisation": "IoT, faible mémoire",
        "Avantage": "Rapide et léger"
    },
    {
        "Niveau": "ML-KEM-768",
        "Utilisation": "Web, Mobile, usage général",
        "Avantage": "Bon équilibre"
    },
    {
        "Niveau": "ML-KEM-1024",
        "Utilisation": "Banque, Cloud, données critiques",
        "Avantage": "Sécurité maximale"
    }
])

st.dataframe(df_kem, use_container_width=True)


# ============================================================
# SESSIONS PRÉCÉDENTES
# ============================================================

st.markdown("---")
afficher_sessions_precedentes()


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        HPQS-AI — Dashboard Streamlit | Module 5 | Salma A + Nabila B
    </div>
    """,
    unsafe_allow_html=True
)
