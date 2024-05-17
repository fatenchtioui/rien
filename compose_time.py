
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

def test_stationarity(timeseries, window):
    # Calcul de la moyenne mobile et de l'écart-type mobile
    rolmean = timeseries.rolling(window).mean()
    rolstd = timeseries.rolling(window).std()

    # Création de la figure et des sous-graphiques
    fig, ax = plt.subplots()

    # Affichage des statistiques mobiles
    ax.plot(timeseries, color='blue', label='Données originales')
    ax.plot(rolmean, color='red', label='Moyenne mobile')
    ax.plot(rolstd, color='black', label='Écart-type mobile')
    ax.legend(loc='best')
    ax.set_title('Moyenne mobile et écart-type mobile')

    # Affichage du graphique dans Streamlit
    st.pyplot(fig)

    # Test de Dickey-Fuller augmenté
    st.subheader('Résultats du test de Dickey-Fuller augmenté :')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Statistique du test', 'Valeur p', '#Lags utilisés', 'Nombre d\'observations utilisées'])
    for key, value in dftest[4].items():
        dfoutput['Valeur critique (%s)' % key] = value
    st.write(dfoutput)


def plot_acf_pacf(data):
    """
    Plot Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) for the time series data.

    Args:
    data (pandas.Series): Time series data.

    Returns:
    matplotlib.figure.Figure: ACF and PACF plots.
    """
    # Plot ACF and PACF
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(data, ax=ax[0], lags=30)
    ax[0].set_title('Autocorrelation Function (ACF)')
    plot_pacf(data, ax=ax[1], lags=30)
    ax[1].set_title('Partial Autocorrelation Function (PACF)')
    return fig

def seasonal_decomposition(data, period):
    """
    Perform seasonal decomposition of time series data.

    Args:
    data (pandas.Series): Time series data.
    period (int): Period of the seasonality.

    Returns:
    tuple: Tuple containing trend, seasonal, and residual components.
    """
    decomposition = STL(data, period=period).fit()
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    return trend, seasonal, residual

def plot_seasonal_decomposition(trend, seasonal, residual, period):
    """
    Plot the components of seasonal decomposition.

    Args:
    trend (pandas.Series): Trend component.
    seasonal (pandas.Series): Seasonal component.
    residual (pandas.Series): Residual component.
    period (int): Period of the seasonality.
    """
    fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    ax[0].plot(trend)
    ax[0].set_title('Trend')
    ax[1].plot(seasonal)
    ax[1].set_title('Seasonal (Period={})'.format(period))
    ax[2].plot(residual)
    ax[2].set_title('Residual')
    ax[2].set_xlabel('Time')
    plt.tight_layout()
    return fig

from statsmodels.tsa.seasonal import STL

def seasonal_decomposition(data, period):
    """
    Perform seasonal decomposition of time series data.

    Args:
    data (pandas.Series): Time series data.
    period (int): Period of the seasonality.

    Returns:
    tuple: Tuple containing trend, seasonal, and residual components.
    """
    decomposition = STL(data, period=period).fit()
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    return trend, seasonal, residual

def plot_seasonal_decomposition(trend, seasonal, residual, period):
    """
    Plot the components of seasonal decomposition.

    Args:
    trend (pandas.Series): Trend component.
    seasonal (pandas.Series): Seasonal component.
    residual (pandas.Series): Residual component.
    period (int): Period of the seasonality.
    """
    fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    ax[0].plot(trend)
    ax[0].set_title('Trend')
    ax[1].plot(seasonal)
    ax[1].set_title('Seasonal (Period={})'.format(period))
    ax[2].plot(residual)
    ax[2].set_title('Residual')
    ax[2].set_xlabel('Time')
    plt.tight_layout()
    return fig
