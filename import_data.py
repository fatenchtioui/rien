import streamlit as st
import pandas as pd


# Fonction pour nettoyer les données
def clean_data(df):
    # Supprimer les valeurs manquantes et les caractères indésirables
    df = df.dropna()
    # Nettoyer les caractères spéciaux
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    # Supprimer les caractères %
    df = df.apply(lambda x: x.str.replace('%', '') if x.dtype == "object" else x)
    #df.to_excel("data.csv")
    return df


# Fonction pour transformer les données en mois par groupe
def transform_data(df):
    transformed_df = df.groupby(['Month', 'Date']).agg({
        'NH Budget': 'sum',
        'NH Actual': 'sum',
        'Sales Bud': 'sum',
        'Sales Act ': 'sum',
        'CLIENT FORCAST S1': 'sum',
        'Production Calendar': 'sum',
        'Customer Calendar': 'sum',
        'Customer Consumption Last 12 week': 'sum',
        'Stock Plant : TIC Tool': 'sum',
        'HC DIRECT': 'mean',
        'HC INDIRECT': 'mean',
        'ABS P': 'mean',
        'ABS NP': 'mean',
        'FLUCTUATION': 'mean'
    }).reset_index() 
    transformed_df.to_csv('df_month.csv')
    return transformed_df

# Fonction pour exporter les données
def export_data(df, file_name):
    df.to_csv(file_name, index=False)
