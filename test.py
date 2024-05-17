import streamlit as st
import pandas as pd
from prophet import Prophet
from sklearn.metrics import r2_score, mean_squared_error
import plotly.graph_objects as go
import os
import joblib

def check_date_consistency(actual_data, predicted_data):
    actual_start_date = actual_data['ds'].min()
    actual_end_date = actual_data['ds'].max()
    predicted_start_date = predicted_data['ds'].min()
    predicted_end_date = predicted_data['ds'].max()

    if actual_start_date != predicted_start_date or actual_end_date != predicted_end_date:
        st.warning("Les plages de dates des données réelles et des prédictions ne correspondent pas.")
    else:
        st.success("Les plages de dates des données réelles et des prédictions correspondent.")

def run_prophet_model(df):
    model = Prophet()
    df = df.rename(columns={"Date": "ds", "NH Actual": "y"})
    model.fit(df)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    return forecast
# Fonction pour charger les modèles de machine learning
def load_ml_models():
    global prophet_model
    prophet_model_path = "model_ex/prophet_model.pkl"
    if os.path.exists(prophet_model_path):
        prophet_model = joblib.load(prophet_model_path)
    else:
        st.error("Prophet model file not found!")


def prepare_data_for_prophet(df):
    df_prophet = df[['Date', 'NH Actual']]
    df_prophet = df_prophet.rename(columns={"Date": "ds", "NH Actual": "y"})
        # If 'ds' column is not present, set 'Date' as 'ds'
    if 'ds' not in df.columns:
        df['ds'] = df['Date']  
    return df_prophet        
def show_ml_simulation_form(df, forecast):
    st.subheader("Simulation with Machine Learning Models")

    # Filter by date range
    min_date = df['Date'].min()
    max_date = df['Date'].max()

    selected_start_date = st.date_input("Select start date", min_value=min_date.date(), max_value=max_date.date(), value=min_date.date())
    selected_end_date = st.date_input("Select end date", min_value=min_date.date(), max_value=max_date.date(), value=max_date.date())

    if selected_start_date and selected_end_date:
        selected_start_date = pd.to_datetime(selected_start_date)
        selected_end_date = pd.to_datetime(selected_end_date)
        df_filtered = df[(df['Date'] >= selected_start_date) & (df['Date'] <= selected_end_date)]
        forecast_filtered = forecast[(forecast['ds'] >= selected_start_date) & (forecast['ds'] <= selected_end_date)]

        # Display input fields for feature values
        feature_names = ["NH Budget","Sales Bud","Sales Act "," Sales Actual/Budget","NH Actual/Budget","Production Calendar","Customer Calendar","ADC Calendar","Customer Consumption Last 12 week","Stock Plant : TIC Tool","CLIENT FORCAST S1","HC DIRECT","HC INDIRECT","ABS P","ABS NP","FLUCTUATION"]
        user_input = {}  # Dictionary to store user input for each feature
        for feature in feature_names:
            if feature in df_filtered.columns:
                user_input[feature] = st.text_input(f"Enter value for {feature}", value=str(df_filtered.iloc[0][feature]))
                max_value = float(df[feature].mean() * 5)  # Convertir en float
                step = float(max_value / 100)  # Convertir en float
                user_input[feature] = st.slider(f"Enter value for {feature}", min_value=0.0, max_value=max_value, value=float(df_filtered.iloc[0][feature]), step=step)
            else:
                user_input[feature] = st.text_input(f"Enter value for {feature}", value='0')  # Replace '0' with an appropriate default value

        # Convert user input to DataFrame
        user_input_df = pd.DataFrame([user_input])

        # Check date consistency
        check_date_consistency(df_filtered, forecast_filtered)

        # Plot results
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_filtered['Date'], y=df_filtered['NH Actual'], mode='lines', name='Actual'))  # Actual data
        fig.add_trace(go.Scatter(x=forecast_filtered['ds'], y=forecast_filtered['yhat'], mode='lines', name='Predicted'))  # Predicted data
        fig.update_layout(title='Actual vs Forecasted NH Actual', xaxis_title='Date', yaxis_title='NH Actual')
        st.plotly_chart(fig)

        # Calculate metrics
        r2 = r2_score(df_filtered['NH Actual'], forecast_filtered['yhat'])
        mse = mean_squared_error(df_filtered['NH Actual'], forecast_filtered['yhat'])
        rmse = mse ** 0.5

        # Display metrics
        st.write(f"R²: {r2}")
        st.write(f"RMSE: {rmse}")
        st.write(f"MSE: {mse}")
