import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model

#def load_model_from_file(model_file):
    #if not model_file:
        #return None
    #try:
        #model = load_model(model_file)
        #return model
    #except Exception as e:
        #st.error(f"Error loading model from file: {e}")
        #return None


def predict_with_model(model, data):
    # Faire des prédictions avec le modèle
    predictions = model.predict(data)
    return predictions