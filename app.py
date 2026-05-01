import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Configuration de la page
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Affichage du logo dans la barre latérale
try:
    st.sidebar.image('téléchargement.png', use_container_width=True)
except:
    pass

# Style CSS personnalisé pour un look premium
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #0d6efd;
        color: white;
        font-weight: bold;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .risk-high {
        color: #dc3545;
        font-size: 24px;
        font-weight: bold;
    }
    .risk-low {
        color: #198754;
        font-size: 24px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Charger le modèle
@st.cache_resource
def load_model():
    return joblib.load('churn_model.pkl')

try:
    model = load_model()
except:
    st.error("⚠️ Modèle non trouvé. Veuillez exécuter 'train_model.py' d'abord.")
    st.stop()

# Header
st.title("📡 Telco Customer Churn Predictor")
st.markdown("---")

# Sidebar pour les entrées
st.sidebar.header("📋 Profil du Client")

def user_input_features():
    # Variables numériques
    st.sidebar.subheader("Données Chiffrées")
    tenure = st.sidebar.slider("Ancienneté (mois)", 0, 72, 12)
    monthly_charges = st.sidebar.slider("Frais mensuels ($)", 18, 118, 50)
    total_charges = st.sidebar.number_input("Frais totaux ($)", 0.0, 9000.0, 500.0)
    
    # Variables catégorielles
    st.sidebar.subheader("Services & Contrat")
    gender = st.sidebar.selectbox("Genre", ("Male", "Female"))
    senior = st.sidebar.selectbox("Sénior", (0, 1))
    partner = st.sidebar.selectbox("Partenaire", ("Yes", "No"))
    dependents = st.sidebar.selectbox("Personnes à charge", ("Yes", "No"))
    phone = st.sidebar.selectbox("Service Téléphonique", ("Yes", "No"))
    multiple_lines = st.sidebar.selectbox("Lignes multiples", ("Yes", "No", "No phone service"))
    internet = st.sidebar.selectbox("Service Internet", ("DSL", "Fiber optic", "No"))
    security = st.sidebar.selectbox("Sécurité en ligne", ("Yes", "No", "No internet service"))
    backup = st.sidebar.selectbox("Backup en ligne", ("Yes", "No", "No internet service"))
    protection = st.sidebar.selectbox("Protection appareil", ("Yes", "No", "No internet service"))
    support = st.sidebar.selectbox("Support technique", ("Yes", "No", "No internet service"))
    tv = st.sidebar.selectbox("Streaming TV", ("Yes", "No", "No internet service"))
    movies = st.sidebar.selectbox("Streaming Movies", ("Yes", "No", "No internet service"))
    contract = st.sidebar.selectbox("Type de contrat", ("Month-to-month", "One year", "Two year"))
    paperless = st.sidebar.selectbox("Facturation sans papier", ("Yes", "No"))
    payment = st.sidebar.selectbox("Mode de paiement", ("Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"))

    data = {
        'gender': gender,
        'SeniorCitizen': senior,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone,
        'MultipleLines': multiple_lines,
        'InternetService': internet,
        'OnlineSecurity': security,
        'OnlineBackup': backup,
        'DeviceProtection': protection,
        'TechSupport': support,
        'StreamingTV': tv,
        'StreamingMovies': movies,
        'Contract': contract,
        'PaperlessBilling': paperless,
        'PaymentMethod': payment,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Affichage des données saisies
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🔍 Analyse des données client")
    st.write("Voici les caractéristiques sélectionnées :")
    st.dataframe(input_df)

    # Prédiction
    prediction_proba = model.predict_proba(input_df)[0][1]
    prediction = 1 if prediction_proba > 0.5 else 0

    st.markdown("---")
    st.subheader("📊 Résultat de la Prédiction")
    
    if prediction == 1:
        st.markdown(f"""
            <div class="prediction-card">
                <h3>Risque de départ : <span class="risk-high">ALERTE (Churn probable)</span></h3>
                <p>Probabilité estimée : {prediction_proba:.2%}</p>
            </div>
        """, unsafe_allow_html=True)
        st.warning("⚠️ Action recommandée : Contacter le client pour une offre de fidélisation.")
    else:
        st.markdown(f"""
            <div class="prediction-card">
                <h3>Risque de départ : <span class="risk-low">FAIBLE (Fidèle)</span></h3>
                <p>Probabilité estimée : {prediction_proba:.2%}</p>
            </div>
        """, unsafe_allow_html=True)
        st.success("✅ Le client semble satisfait de ses services actuels.")

with col2:
    st.subheader("📊 Pourquoi le modèle dit ça ?")
    try:
        importance_df = pd.read_csv('feature_importance.csv')
        st.bar_chart(importance_df.set_index('Feature'))
    except:
        st.info("Importance des variables non disponible.")

    st.subheader("💡 Actions Recommandées")             
    recoms = []
    if input_df['Contract'][0] == 'Month-to-month':
        recoms.append("🎯 **Fidélisation** : Proposer un passage à un contrat annuel.")
    if input_df['InternetService'][0] == 'Fiber optic':
        recoms.append("🌐 **Technique** : Offrir un diagnostic gratuit de la ligne.")
    if input_df['OnlineSecurity'][0] == 'No':
        recoms.append("🛡️ **Upsell** : Suggérer le pack 'Sécurité Totale'.")
    
    if not recoms:
        st.success("Profil client stable.")
    else:
        for r in recoms:
            st.write(r)
    
    st.markdown("---")
    st.write("Précision globale du modèle :")
    st.progress(0.79) 

st.caption("Application développée pour la validation de l'examen de Machine Learning - 2026")
