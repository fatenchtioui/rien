import pandas as pd
import streamlit as st
import pickle
import os
from datetime import timedelta
from sklearn.metrics import r2_score, mean_squared_error
import plotly.graph_objects as go
import numpy as np
import base64

# Charger les données et les traiter si nécessaire
@st.cache  # Cachez le jeu de données pour éviter de le recharger à chaque rerun
def load_data():
    
    df['Date'] = pd.to_datetime(df['Date'])
    df.fillna(0, inplace=True)  # Remplir les valeurs manquantes avec 0 (vous pouvez ajuster selon vos besoins)
    return df

# Charger le modèle sauvegardé
def load_model(filename):
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

# Filtrer les données en fonction de la plage de dates sélectionnée
def filter_data(df, date_range):
    return df[(df['Date'] >= date_range[0]) & (df['Date'] <= date_range[1])]

# Sélectionner les colonnes de fonctionnalités à modifier
def select_feature_columns(df):
    return st.multiselect("Sélectionner les variables de fonctionnalités à modifier:", df.columns)

# Afficher les données filtrées
def display_filtered_data(filtered_data):
    st.write(filtered_data)

# Faire des prédictions pour la dernière date disponible dans les données filtrées
def predict_last_date(filtered_data, feature_columns, model):
        # Function to make predictions
    def predict(model, data):
        return model.predict(data)
    last_date = filtered_data['Date'].max()
    filtered_data_last_date = filtered_data[filtered_data['Date'] == last_date]
    X_last_date = filtered_data_last_date[feature_columns].values
    predictions_last_date = predict(model, X_last_date)
    return predictions_last_date, filtered_data_last_date

# Calculer les métriques d'évaluation
def calculate_metrics(filtered_data, predictions_last_date):
    r2 = r2_score(filtered_data['NH Actual'], predictions_last_date)
    rmse = np.sqrt(mean_squared_error(filtered_data['NH Actual'], predictions_last_date))
    mse = mean_squared_error(filtered_data['NH Actual'], predictions_last_date)
    return r2, rmse, mse

# Afficher les métriques d'évaluation
def display_metrics(r2, rmse, mse):
    st.subheader("Métriques d'évaluation du modèle :")
    st.write(f"R²: {r2}")
    st.write(f"RMSE: {rmse}")
    st.write(f"MSE: {mse}")

# Plot des prédictions
def plot_predictions(filtered_data_last_date, predictions_last_date):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered_data_last_date.index, y=predictions_last_date, mode='lines', name='Prédiction'))
    fig.update_layout(title='Valeurs prédites', xaxis_title='Index', yaxis_title='Valeur prédite')
    st.plotly_chart(fig)

# Combinaison des données filtrées avec les prédictions
def combine_data(filtered_data_last_date, predictions_last_date):
    combined_data = pd.concat([filtered_data_last_date.reset_index(drop=True), pd.DataFrame(predictions_last_date, columns=['Prédiction'])], axis=1)
    return combined_data
st.write(combine_data)
# Fonction pour télécharger les données combinées au format CSV
def download_csv(data, filename):
    csv = data.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">Télécharger {filename}</a>'
    return href
