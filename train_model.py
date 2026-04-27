import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

# 1. Charger les données
data_path = "WA_Fn-UseC_-Telco-Customer-Churn (1).csv"
df = pd.read_csv(data_path)

# 2. Nettoyage des données
# Conversion de TotalCharges en numérique (les espaces deviennent NaN)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# Remplir les NaN par 0 (clients très récents sans facturation totale encore)
df['TotalCharges'] = df['TotalCharges'].fillna(0)

# Supprimer customerID
df = df.drop('customerID', axis=1)

# Définir X et y
X = df.drop('Churn', axis=1)
y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# 3. Séparation des variables
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(exclude=['object']).columns.tolist()

# 4. Pipeline de prétraitement
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

# 5. Pipeline complet avec Random Forest
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 6. Entraînement
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Entraînement du modèle...")
model.fit(X_train, y_train)

# 7. Extraction de l'importance des variables
# Récupérer les noms des colonnes après transformation
ohe_features = model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(cat_cols)
feature_names = num_cols + list(ohe_features)
importances = model.named_steps['classifier'].feature_importances_

# Créer un DataFrame pour les importances
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False).head(10) # Top 10

# 8. Sauvegarde
joblib.dump(model, 'churn_model.pkl')
importance_df.to_csv('feature_importance.csv', index=False)
print("Modèle et importance des variables sauvegardés.")

# Score rapide pour vérification
score = model.score(X_test, y_test)
print(f"Précision du modèle sur le test set : {score:.2f}")
