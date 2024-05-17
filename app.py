import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import joblib
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from analyse_data import  profile_data
from predict import load_model,filter_data,select_feature_columns,display_filtered_data,predict_last_date,calculate_metrics,display_metrics,plot_predictions,combine_data,download_csv
from compose_time import plot_acf_pacf,test_stationarity,seasonal_decomposition, plot_seasonal_decomposition
from prophet_page import get_prophet
import math
import os
import plotly.graph_objects as go
from ml_page import get_random_forest_model,get_br,get_gb
from metrics import calculate_metrics
from deep_page import get_ann, get_cnn_model, get_dnn_model, get_lstm, get_mlp_model
from simulation_page import predict_with_model
from tensorflow.keras.models import load_model
import tensorflow as tf
from prophet import Prophet
from test import show_ml_simulation_form, load_ml_models,run_prophet_model,prepare_data_for_prophet
import datetime
from sklearn.ensemble import GradientBoostingRegressor


model_ml={
    'Random Forest': get_random_forest_model,
    'Gradient Boosting': get_gb,
    'Bagging Regressor': get_br,     
}
models={
    'DNN': get_dnn_model,
    'MLP': get_mlp_model,
    'CNN': get_cnn_model,
    'LSTM': get_lstm,
    'ANN': get_ann     
}

    
    # Sidebar navigation
with st.sidebar:
        st.title("PACO App / Norm_Hour Back_Testing_System")
        st.image("C:\\Users\\faten\\OneDrive\\Bureau\\rien\\leoni.png")
        st.title('Model selection')
        choice = st.radio("Navigation", ["Data Exploration", "Time Series Analysis", "Seasonal Decomposition","Prophet Simulator","ML Simulator","Deep Learning Simulator","Simulation with Prophet","Simulation with ML","Simulation with Gradient Boosting Regressor","Simulation with Deep Learning","Prediction"])
        st.info('This application shows the results of all models used for predicting NH Actual')

    # Charger les données si un fichier a été téléchargé
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv","xlsx"])
if uploaded_file is not None:
      df = pd.read_excel(uploaded_file)
else:
       df = None




if choice == "Data Exploration":
    st.title("Data Analysis")
    if df is not None:
        profile_data(df)
    else:
        st.warning("Please upload a CSV file to proceed.")
       

elif choice == "Time Series Analysis":
    # Composant interactif pour obtenir la taille de la fenêtre pour la moyenne mobile
    window_size = st.slider("Enter the window size for moving average:", min_value=2, max_value=len(df), value=10)

    # Extraction de la série temporelle du DataFrame
    timeseries_column = st.sidebar.selectbox("Select a column for stationarity test", df.columns)
    if timeseries_column:
        if timeseries_column in df.columns:
            timeseries = df[timeseries_column].dropna()
            # Vérification de la validité de la série temporelle
            if not timeseries.empty:
                # Calcul de la stationnarité de la série temporelle
                test_stationarity(timeseries, window=window_size) 
                # Plot ACF and PACF
                st.subheader("Autocorrelation and Partial Autocorrelation Plot")
                fig = plot_acf_pacf(timeseries)
                st.pyplot(fig)
            else:
                st.warning("Selected column is empty.")
        else:
            st.warning("Selected column does not exist in the DataFrame.")
    else:
        st.warning("Please select a column for the stationarity test.")

elif choice == "Seasonal Decomposition":
        
            # Date filter
            st.sidebar.subheader("Date Filter")
            start_date = st.sidebar.date_input("Start Date")
            end_date = st.sidebar.date_input("End Date")

            # Perform seasonal decomposition
            st.sidebar.subheader("Seasonal Decomposition")
            column_for_decomposition = st.sidebar.selectbox("Select a column for seasonal decomposition",
                                                            df.columns)
            seasonal_period = st.sidebar.number_input("Enter the period of seasonality", min_value=1, max_value=365,
                                                      step=1, value=7)
            if st.sidebar.button("Perform Seasonal Decomposition"):
                if column_for_decomposition:
                    trend, seasonal, residual = seasonal_decomposition(df[column_for_decomposition],
                                                                       seasonal_period)
                    st.subheader("Seasonal Decomposition Results")
                    st.write("Trend Component:")
                    st.write(trend)
                    st.write("Seasonal Component (Period={}):".format(seasonal_period))
                    st.write(seasonal)
                    st.write("Residual Component:")
                    st.write(residual)

                    # Plot seasonal decomposition
                    st.subheader("Plot of Seasonal Decomposition")
                    fig = plot_seasonal_decomposition(trend, seasonal, residual, seasonal_period)
                    st.pyplot(fig)
                else:
                    st.warning("Please select a column for seasonal decomposition.")

   


elif choice == "Prophet Simulator":
     df.fillna(0, inplace=True)
     df.replace('%', '', regex=True, inplace=True)
     st.write(df)
     if 'ds' not in df.columns:
         df['ds'] = df['Date']  # Remplacez 'Nom_de_la_colonne_Date' par le nom correct de la colonne contenant les dates

        # Appeler la fonction pour entraîner le modèle Prophet et faire des prédictions
         model = get_prophet(df)
         predictions = model.predict(df)  # Faire des prédictions
                    # Separate features and target for training and validation
         feature_columns =st.multiselect("Select feature variables:", df.columns)
         target_column = ['NH Actual']
         train_size = int(0.85 * len(df))
         multivariate_df = df[['Date'] + target_column + feature_columns].copy()
         multivariate_df.columns = ['ds', 'y'] + feature_columns

         train = multivariate_df.iloc[:train_size, :]
         valid = multivariate_df.iloc[train_size:, :]
    

         # Separate features and target for training and validation
         x_train = train.drop(columns=['y'])
         y_train = train['y']
         x_valid = valid.drop(columns=['y'])
         y_valid = valid['y']
                # Afficher les métriques des modèles
                #st.title("Model Metrics")
                #r2, rmse, mse = calculate_metrics(model, x_valid, y_valid)
                #metrics_df = pd.DataFrame({'Model': [prediction_algorithm], 'R²': [r2], 'RMSE': [rmse], 'MSE': [mse]})
                #st.table(metrics_df)
                        # Extract validation target values
         y_valid = df['NH Actual'][-len(predictions):]
 # Assuming the last predictions correspond to the validation set

                # Display metrics
         score_mae_valid = mean_absolute_error(y_valid, predictions['yhat'])
         score_rmse_valid = math.sqrt(mean_squared_error(y_valid, predictions['yhat']))
         score_r2_valid = r2_score(y_valid, predictions['yhat'])

         st.write(f"Mean Absolute Error: {score_mae_valid}")
         st.write(f"Root Mean Squared Error: {score_rmse_valid}")
         st.write(f"R-squared: {score_r2_valid}")
                # Créer un graphique
         fig = go.Figure()
         fig.add_trace(go.Scatter(x=df['Date'], y=df['NH Actual'], mode='lines', name='Actual'))
         fig.add_trace(go.Scatter(x=df['Date'], y=predictions['yhat'], mode='lines', name='Predicted'))

                # Mise en forme du graphique
         fig.update_layout(title='NH Actual vs Predicted', xaxis_title='Date', yaxis_title='NH Actual')
                     # Afficher le graphique
         st.plotly_chart(fig)

                # Bouton de téléchargement pour les valeurs prédites
         #if st.button("Download Predicted Data as CSV"):
                    # Télécharger les valeurs prédites avec les features
                    # Assurez-vous de remplacer 'predicted_data.csv' par le nom de fichier souhaité
                    #predicted_df = predictions[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'Predicted'})
                    #st.download_button(label="Download Predicted Data", data=predicted_df.to_csv(index=False), file_name='predicted_data.csv', mime='text/csv')
          
           # Enregistrement automatique du modèle après l'entraînement
         model_dir = "model_ex"
         if not os.path.exists(model_dir):
            os.makedirs(model_dir)
         filepath = os.path.join(model_dir, 'prophet_model.pkl')
         with open(filepath, 'wb') as f:
            pickle.dump(model, f)

         st.success("Model extracted successfully!")  

elif choice == "ML Simulator":
    df.fillna(0, inplace=True)
    df.replace('%', '', regex=True, inplace=True)
    st.write(df)

    # Afficher les options pour choisir l'algorithme de prédiction
    st.subheader("Choose Prediction Algorithm")
    prediction_algorithm = st.selectbox("Select Prediction Algorithm", ["Random Forest", "Gradient Boosting", "Bagging Regressor"])
    train_size = int(0.85 * len(df))
    train_data = df.iloc[:train_size] 
    feature_columns = st.multiselect("Select feature variables:", df.columns)
    target_column = ['NH Actual']  # Assurez-vous que c'est une liste
   
     # Charger et exécuter le modèle d'apprentissage automatique approprié
    model = model_ml[prediction_algorithm](train_data, feature_columns, target_column)  # Utilisez les parenthèses pour appeler la fonction

     # Séparation des données d'entraînement et de validation
    train_size = int(0.85 * len(df))
    train_data = df.iloc[:train_size]
    valid_data = df.iloc[train_size:]

    # Faites des prédictions avec le modèle
    X_valid = valid_data[feature_columns]  # Utilisez directement feature_columns sans les parenthèses
    y_valid = valid_data[target_column]

    # Entraîner le modèle sur les données d'entraînement
    model.fit(train_data[feature_columns], train_data[target_column])  # Utilisez directement feature_columns sans les parenthèses

    # Faire des prédictions avec le modèle entraîné
    predictions = model.predict(X_valid)

    # Afficher les métriques des modèles
    st.title("Model Metrics")
    r2, rmse, mse = calculate_metrics(model, X_valid, y_valid)  # Passer les vraies valeurs (y_valid)
    metrics_df = pd.DataFrame({'Model': [prediction_algorithm], 'R²': [r2], 'RMSE': [rmse], 'MSE': [mse]})
    st.table(metrics_df)




    # Créer un graphique
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=valid_data['Date'], y=valid_data['NH Actual'], mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=valid_data['Date'], y=predictions, mode='lines', name='Predicted'))

    # Mise en forme du graphique
    fig.update_layout(title='NH Actual vs Predicted', xaxis_title='Date', yaxis_title='NH Actual')

    # Afficher le graphique
    st.plotly_chart(fig)
        # Enregistrement automatique du modèle après l'exécution
    model_dir = "model_ex"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filepath = os.path.join(model_dir, f"{prediction_algorithm.lower()}_model.pkl")
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

    st.success("Model extracted successfully!")










elif choice == "Deep Learning Simulator":
    df.fillna(0, inplace=True)
    df.replace('%', '', regex=True, inplace=True)
    st.write(df)

    # Afficher les options pour choisir l'algorithme de prédiction
    st.subheader("Choose Prediction Algorithm")
    prediction_algorithm = st.selectbox("Select Prediction Algorithm", ["DNN", "CNN", "MLP","LSTM","ANN"])
    train_size = int(0.85 * len(df))
    train_data = df.iloc[:train_size] 
    feature_columns = st.multiselect("Select feature variables:", df.columns)
    target_column = ['NH Actual']  # Assurez-vous que c'est une liste
   
     # Charger et exécuter le modèle d'apprentissage automatique approprié
    model = models[prediction_algorithm](train_data, feature_columns, target_column)  # Utilisez les parenthèses pour appeler la fonction

     # Séparation des données d'entraînement et de validation
    train_size = int(0.85 * len(df))
    train_data = df.iloc[:train_size]
    valid_data = df.iloc[train_size:]

    # Faites des prédictions avec le modèle
    X_valid = valid_data[feature_columns]  # Utilisez directement feature_columns sans les parenthèses
    y_valid = valid_data[target_column]


    # Faire des prédictions avec le modèle entraîné
    predictions = model.predict(X_valid)

    # Afficher les métriques des modèles
    st.title("Model Metrics")
    r2, rmse, mse = calculate_metrics(model, X_valid, y_valid)  # Passer les vraies valeurs (y_valid)
    metrics_df = pd.DataFrame({'Model': [prediction_algorithm], 'R²': [r2], 'RMSE': [rmse], 'MSE': [mse]})
    st.table(metrics_df)




    # Créer un graphique
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=valid_data['Date'], y=valid_data['NH Actual'], mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=valid_data['Date'], y=predictions, mode='lines', name='Predicted'))

    # Mise en forme du graphique
    fig.update_layout(title='NH Actual vs Predicted', xaxis_title='Date', yaxis_title='NH Actual')

    # Afficher le graphique
    st.plotly_chart(fig)
        # Enregistrement automatique du modèle après l'entraînement
    model_dir = "model_ex"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model.save(os.path.join(model_dir, f'{prediction_algorithm}_model.h5'))

    st.success("Model extracted successfully!")
elif choice == "Simulation with Prophet":

    st.title("prophet Simulator")

    # Charger les modèles au démarrage de l'application
    load_ml_models()
        # Charger et entraîner le modèle Prophet
    forecast = run_prophet_model(prepare_data_for_prophet(df))

    # Afficher le formulaire de simulation
    show_ml_simulation_form(df, forecast)
    import io
    import base64
    import pandas as pd
    combined_data = pd.concat([df.reset_index(drop=True), pd.DataFrame(forecast, columns=['Predicted'])], axis=1)

    # Function to convert DataFrame to CSV file and create a download link
    def download_csv(data, filename):
        csv = data.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # Encode as base64
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">Download {filename}</a>'
        return href



          
    # Add button to download combined data as CSV
    st.markdown(download_csv(combined_data, "combined_data"), unsafe_allow_html=True)






    #show_simulation_form(df, forecast)
elif choice == "Simulation with ML":

    df.fillna(0, inplace=True)
    
    # Load the saved model
    def load_model(filename):
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        return model

    # Function to make predictions
    def predict(model, data):
        return model.predict(data)

    # Load the dataset
    @st.cache  # Cache the dataset to avoid reloading on every rerun
    def load_data(filename):
        return pd.read_csv(filename)


    # Filter data based on user input
    filtered_data = df.copy()  # Initialize with full data
    if st.checkbox("Enable Data Filtering"):
        # Add filter options here
        # For example, filter by date
        date_range = st.date_input("Select Date Range", [df['Date'].min(), df['Date'].max()])
        date_range = [pd.Timestamp(date) for date in date_range]  # Convert to Timestamp objects
        filtered_data = df[(df['Date'] >= date_range[0]) & (df['Date'] <= date_range[1])]
    # Add more filters as needed

    # Display the filtered data
    st.write(filtered_data)

    # Load the model
    model_filename = st.text_input("Enter model file path:", "model.pkl")
    if os.path.exists(model_filename):
        model = load_model(model_filename)
    else:
        st.warning("Model file not found. Please provide a valid file path.")

    feature_columns = st.multiselect("Select feature variables to modify:", df.columns)
    for column in feature_columns:
        max_value = float(df[column].mean() * 5)  # Convertir en float
        new_value = st.number_input(f"Enter new value for {column}:", value=df[column].mean())
        step = float(max_value / 100)  # Convertir en float
        new_value = st.slider(f"Enter new value for {column}:", min_value=0.0, max_value=max_value, value=new_value, step=step)
        filtered_data[column] = new_value
    target_column = 'NH Actual'
    # Make predictions
    if st.button("Predict"):
        # Assuming model is a BaggingRegressor with DecisionTreeRegressor as base estimator
        feature_importances = np.mean([tree.feature_importances_ for tree in model.estimators_], axis=0)

        # Sort the features based on their importances
        feature_indices = np.argsort(feature_importances)[::-1]
        X = df[feature_columns].values
        # Convert X to a numpy array
        X = np.array(X)

        # Index X based on the sorted feature indices
        X = X[:, feature_indices]
        # Check if X is empty
        if X.size==0:
            st.warning("No data selected for prediction. Please select at least one feature variable.")
        else:
            # Make predictions only if X is not empty
            predictions = predict(model, X)

            # Display predictions
            st.subheader("Predictions")
            st.write(predictions)



            # Plot predictions
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=filtered_data.index, y=predictions, mode='lines', name='Predicted'))
            fig.update_layout(title='Predicted Values', xaxis_title='Index', yaxis_title='Predicted Value')
            st.plotly_chart(fig)

            # Calculate evaluation metrics
            y_true = filtered_data[target_column].values.flatten()
            r2 = r2_score(df['NH Actual'], predictions)
            rmse = np.sqrt(mean_squared_error(df['NH Actual'], predictions))
            mse = mean_squared_error(df['NH Actual'], predictions)

            # Display evaluation metrics
            st.subheader("Model Evaluation Metrics")
            st.write(f"R²: {r2}")
            st.write(f"RMSE: {rmse}")
            st.write(f"MSE: {mse}")
            import io
            import base64
            import pandas as pd
            combined_data = pd.concat([filtered_data.reset_index(drop=True), pd.DataFrame(predictions, columns=['Predicted'])], axis=1)

            # Function to convert DataFrame to CSV file and create a download link
            def download_csv(data, filename):
                csv = data.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()  # Encode as base64
                href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">Download {filename}</a>'
                return href



          
            # Add button to download combined data as CSV
            st.markdown(download_csv(combined_data, "combined_data"), unsafe_allow_html=True)


elif choice == "Simulation with Gradient Boosting Regressor":
    df.fillna(0, inplace=True)
    
    # Load the saved model
    def load_model(filename):
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        return model

    # Function to make predictions
    def predict(model, data):
        return model.predict(data)

    # Load the dataset
    @st.cache  # Cache the dataset to avoid reloading on every rerun
    def load_data(filename):
        return pd.read_csv(filename)


    # Filter data based on user input
    filtered_data = df.copy()  # Initialize with full data
    if st.checkbox("Enable Data Filtering"):
        # Add filter options here
        # For example, filter by date
        date_range = st.date_input("Select Date Range", [df['Date'].min(), df['Date'].max()])
        date_range = [pd.Timestamp(date) for date in date_range]  # Convert to Timestamp objects
        filtered_data = df[(df['Date'] >= date_range[0]) & (df['Date'] <= date_range[1])]
    # Add more filters as needed

    # Display the filtered data
    st.write(filtered_data)

    # Load the model
    model_filename = st.text_input("Enter model file path:", "model.pkl")
    if os.path.exists(model_filename):
        model = load_model(model_filename)
    else:
        st.warning("Model file not found. Please provide a valid file path.")

    feature_columns = st.multiselect("Select feature variables to modify:", df.columns)
    for column in feature_columns:
        max_value = float(df[column].mean() * 5)  # Convertir en float
        new_value = st.number_input(f"Enter new value for {column}:", value=df[column].mean())
        step = float(max_value / 100)  # Convertir en float
        new_value = st.slider(f"Enter new value for {column}:", min_value=0.0, max_value=max_value, value=new_value, step=step)
        filtered_data[column] = new_value
    target_column = 'NH Actual'
    # Make predictions
    if st.button("Predict"):
        if isinstance(model, GradientBoostingRegressor):
            feature_importances = model.feature_importances_
        
        # Sort the features based on their importances
        feature_indices = np.argsort(feature_importances)
        X = df[feature_columns].values
        # Convert X to a numpy array
        X = np.array(X)

        # Index X based on the sorted feature indices
        X = X[:, feature_indices]
        # Check if X is empty
        if X.size==0:
            st.warning("No data selected for prediction. Please select at least one feature variable.")
        else:
            # Make predictions only if X is not empty
            predictions = predict(model, X)

            # Display predictions
            st.subheader("Predictions")
            st.write(predictions)



            # Plot predictions
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=filtered_data.index, y=predictions, mode='lines', name='Predicted'))
            fig.update_layout(title='Predicted Values', xaxis_title='Index', yaxis_title='Predicted Value')
            st.plotly_chart(fig)

            # Calculate evaluation metrics
            y_true = filtered_data[target_column].values.flatten()
            r2 = r2_score(df['NH Actual'], predictions)
            rmse = np.sqrt(mean_squared_error(df['NH Actual'], predictions))
            mse = mean_squared_error(df['NH Actual'], predictions)

            # Display evaluation metrics
            st.subheader("Model Evaluation Metrics")
            st.write(f"R²: {r2}")
            st.write(f"RMSE: {rmse}")
            st.write(f"MSE: {mse}")
            import io
            import base64
            import pandas as pd
            combined_data = pd.concat([filtered_data.reset_index(drop=True), pd.DataFrame(predictions, columns=['Predicted'])], axis=1)

            # Function to convert DataFrame to CSV file and create a download link
            def download_csv(data, filename):
                csv = data.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()  # Encode as base64
                href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">Download {filename}</a>'
                return href



          
            # Add button to download combined data as CSV
            st.markdown(download_csv(combined_data, "combined_data"), unsafe_allow_html=True)




elif choice == "Simulation with Deep Learning":
    df.fillna(0, inplace=True)

    # Load the saved model
    def load_dl_model(model_path):
        try:
            model = load_model(model_path)
            return model
        except Exception as e:
            print("Error loading the model:", e)
            return None

    # Function to make predictions
    def predict(model, data):
        return model.predict(data)

    # Filter data based on user input
    filtered_data = df.copy()  # Initialize with full data
    if st.checkbox("Enable Data Filtering"):
        # Add filter options here
        # For example, filter by date
        date_range = st.date_input("Select Date Range", [df['Date'].min(), df['Date'].max()])
        date_range = [pd.Timestamp(date) for date in date_range]  # Convert to Timestamp objects
        filtered_data = df[(df['Date'] >= date_range[0]) & (df['Date'] <= date_range[1])]

    # Display the filtered data
    st.write(filtered_data)

    # Load the model
    model_choice = st.selectbox("Select Model", ["ANN", "DNN", "CNN", "LSTM", "MLP"])
    model_path_mapping = {
        "ANN": "model_ex/ANN_model.h5",
        "DNN": "model_ex/DNN_model.h5",
        "CNN": "model_ex/CNN_model.h5",
        "LSTM": "model_ex/LSTM_model.h5",
        "MLP": "model_ex/MLP_model.h5"
    }
    model_path = model_path_mapping.get(model_choice)

    # Debugging: Print the absolute path of the model file
    st.write("Absolute model file path:", os.path.abspath(model_path))

    if model_path:
        if os.path.exists(model_path):
            try:
                model = load_dl_model(model_path)
                if model is None:
                    st.warning("Failed to load the model. Please check the model file.")
            except Exception as e:
                st.error(f"An error occurred while loading the model: {e}")
        else:
            st.warning("Model file not found. Please make sure the model file exists.")
    else:
        st.warning("Invalid model choice.")

    feature_columns = st.multiselect("Select feature variables to modify:", df.columns)
    for column in feature_columns:
        max_value = float(df[column].mean() * 5)  # Convertir en float
        new_value = st.number_input(f"Enter new value for {column}:", value=df[column].mean())
        step = float(max_value / 100)  # Convertir en float
        new_value = st.slider(f"Enter new value for {column}:", min_value=0.0, max_value=max_value, value=new_value, step=step)
        filtered_data[column] = new_value

    # Make predictions
    if st.button("Predict"):
        if model is not None:
            # Select features and target
            X = filtered_data[feature_columns].values
            y = filtered_data['NH Actual'].values

            # Make predictions
            predictions = predict(model, X)

            # Display predictions
            st.subheader("Predictions")
            st.write(predictions)

            # Plot predictions
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=filtered_data.index, y=predictions, mode='lines', name='Predicted'))
            fig.update_layout(title='Predicted Values', xaxis_title='Index', yaxis_title='Predicted Value')
            st.plotly_chart(fig)

            # Calculate evaluation metrics
            r2 = r2_score(y, predictions)
            rmse = np.sqrt(mean_squared_error(y, predictions))
            mse = mean_squared_error(y, predictions)

            # Display evaluation metrics
            st.subheader("Model Evaluation Metrics")
            st.write(f"R²: {r2}")
            st.write(f"RMSE: {rmse}")
            st.write(f"MSE: {mse}")
            import io
            import base64
            import pandas as pd
            combined_data = pd.concat([filtered_data.reset_index(drop=True), pd.DataFrame(predictions, columns=['Predicted'])], axis=1)

            # Function to convert DataFrame to CSV file and create a download link
            def download_csv(data, filename,model):
                csv = data.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()  # Encode as base64
                href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">Download {filename} ({model})</a>'
                return href



          
            # Add button to download combined data as CSV
            st.markdown(download_csv(combined_data, "combined_data", model_choice), unsafe_allow_html=True)
       
       
       
       
        else:
            st.error("Failed to load the model. Please check the model file path.")



if choice == "Prediction":
    from datetime import timedelta
    df.fillna(0, inplace=True)
    
    # Load the saved model
    def load_model(filename):
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        return model

    # Function to make predictions
    def predict(model, data):
        return model.predict(data)

    # Load the dataset
    @st.cache  # Cache the dataset to avoid reloading on every rerun
    def load_data(filename):
        return pd.read_csv(filename)


    # Filter data based on user input
    filtered_data = df.copy()  # Initialize with full data
    if st.checkbox("Enable Data Filtering"):
        # Add filter options here
        # For example, filter by date
        date_range = st.date_input("Select Date Range", [df['Date'].min(), df['Date'].max()+timedelta(days=30)])
        date_range = [pd.Timestamp(date) for date in date_range]  # Convert to Timestamp objects
        filtered_data = df[(df['Date'] >= date_range[0]) & (df['Date'] <= date_range[1])]
    # Add more filters as needed

    # Display the filtered data
    st.write(filtered_data)

    # Load the model
    model_filename = st.text_input("Enter model file path:", "model.pkl")
    if os.path.exists(model_filename):
        model = load_model(model_filename)
    else:
        st.warning("Model file not found. Please provide a valid file path.")

    feature_columns = st.multiselect("Select feature variables to modify:", df.columns)
    for column in feature_columns:
        max_value = float(df[column].mean() * 5)  # Convertir en float
        new_value = st.number_input(f"Enter new value for {column}:", value=df[column].mean())
        step = float(max_value / 100)  # Convertir en float
        new_value = st.slider(f"Enter new value for {column}:", min_value=0.0, max_value=max_value, value=new_value, step=step)
        filtered_data[column] = new_value
    target_column = 'NH Actual'
    # Make predictions
    if st.button("Predict"):
        # Assuming model is a BaggingRegressor with DecisionTreeRegressor as base estimator
        feature_importances = np.mean([tree.feature_importances_ for tree in model.estimators_], axis=0)

        # Sort the features based on their importances
        feature_indices = np.argsort(feature_importances)[::-1]
        X = df[feature_columns].values
        # Convert X to a numpy array
        X = np.array(X)

        # Index X based on the sorted feature indices
        X = X[:, feature_indices]
        # Check if X is empty
        if X.size==0:
            st.warning("No data selected for prediction. Please select at least one feature variable.")
        else:
            # Make predictions only if X is not empty
            predictions = predict(model, X)

            # Display predictions
            st.subheader("Predictions")
            st.write(predictions)



            # Plot predictions
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=filtered_data.index, y=predictions, mode='lines', name='Predicted'))
            fig.update_layout(title='Predicted Values', xaxis_title='Index', yaxis_title='Predicted Value')
            st.plotly_chart(fig)

            # Calculate evaluation metrics
            y_true = filtered_data[target_column].values.flatten()
            r2 = r2_score(df['NH Actual'], predictions)
            rmse = np.sqrt(mean_squared_error(df['NH Actual'], predictions))
            mse = mean_squared_error(df['NH Actual'], predictions)

            # Display evaluation metrics
            st.subheader("Model Evaluation Metrics")
            st.write(f"R²: {r2}")
            st.write(f"RMSE: {rmse}")
            st.write(f"MSE: {mse}")
            import io
            import base64
            import pandas as pd
            combined_data =  pd.DataFrame(predictions)
            # Affichage des données combinées
            st.subheader("Prédictions pour les 7 prochains mois")
            st.write(combined_data)

            # Function to convert DataFrame to CSV file and create a download link
            def download_csv(data, filename):
                csv = data.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()  # Encode as base64
                href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">Download {filename}</a>'
                return href



          
            # Add button to download combined data as CSV
            st.markdown(download_csv(combined_data, "combined_data"), unsafe_allow_html=True)
            # Affichage des données combinées
