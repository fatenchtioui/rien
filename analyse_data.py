import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import altair as alt
import altair as alt
import pandas as pd
import numpy as np
import streamlit as st
import base64

import matplotlib.pyplot as plt
def profile_data(df):
    st.subheader("Exploratory Data Analysis")

    # Data cleaning and preprocessing
    st.write("Data Cleaning and Preprocessing:")
    st.write("- Converting 'Date' column to datetime")
    try:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    except ValueError:
        st.warning("Some dates could not be converted to datetime.")

    # Filter by date range
    min_date = df['Date'].min()
    max_date = df['Date'].max()

    # Determine a default value for the start date
    default_start_date = min_date + (max_date - min_date) // 3
    default_end_date = max_date - (max_date - min_date) // 3

    selected_start_date = st.date_input("Select start date", min_value=min_date.date(), max_value=max_date.date(), value=default_start_date.date())
    selected_end_date = st.date_input("Select end date", min_value=min_date.date(), max_value=max_date.date(), value=default_end_date.date())

    if selected_start_date and selected_end_date:
        selected_start_date = pd.to_datetime(selected_start_date)
        selected_end_date = pd.to_datetime(selected_end_date)
        df = df[(df['Date'] >= selected_start_date) & (df['Date'] <= selected_end_date)]
    else:
        st.warning("Please select a valid date range.")

    # Filter by selected years
    if 'Date' in df.columns:
        df['Year'] = df['Date'].dt.year
    selected_years = st.multiselect("Select years", df['Year'].dropna().unique())
    if selected_years:
        df = df[df['Year'].isin(selected_years)]
    else:
        st.warning("Please select at least one year.")
    
    # Select columns for visualization
    selected_columns = st.multiselect("Select columns for visualization", df.columns)
    if not selected_columns:
        st.warning("Please select at least one column for visualization.")
        return

    # Plot data
    if 'Date' in df.columns:
        chart = alt.Chart(df).mark_line().encode(
            x='Date:T',
            y=alt.Y(selected_columns[0], type='quantitative'),
            color=alt.Color('Year:N')
        ).properties(
            width=800,
            height=500
        )
        st.subheader("Plot")
        st.altair_chart(chart, use_container_width=True)
    else:
        st.warning("No 'Date' column found for plotting.")

    # Display basic DataFrame information
    st.subheader("DataFrame Information")
    st.write("Number of Rows:", df.shape[0])
    st.write("Number of Columns:", df.shape[1])
    st.write("Column Names:", df.columns.tolist())
    st.write("Data Description:")
    st.write(df.describe())
    st.write("DataFrame Info:")
    st.write(df.info())

    # Correlation matrix
    correlation_matrix = df.corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title("Lower triangle of the correlation matrix")
    st.pyplot(plt)
    correlation_matrix.to_csv('correlation_matrix.csv', index=False)
    st.subheader("Correlation Matrix")
    st.write(correlation_matrix)
