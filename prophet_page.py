import pandas as pd
from prophet import Prophet

def get_prophet(df):
    # Feature columns and target column
    feature_columns = [
        'NH Budget', 'CLIENT FORCAST S1',
        'Production Calendar', 'Customer Calendar',
        'Customer Consumption Last 12 week', 'Stock Plant : TIC Tool',
        'HC DIRECT', 'HC INDIRECT', 'ABS P', 'ABS NP', 'FLUCTUATION'
    ]
    target_column = ['NH Actual']

    # Train/validation split
    train_size = int(0.85 * len(df))
    multivariate_df = df[['Date'] + target_column + feature_columns].copy()
    multivariate_df.columns = ['ds', 'y'] + feature_columns

    train = multivariate_df.iloc[:train_size, :]

    # Separate features and target for training
    x_train = train.drop(columns=['y'])
    y_train = train['y']

    # Train the model
    model = Prophet()
    for column in feature_columns:
        model.add_regressor(column)

    # Fit the model with training set
    model.fit(train)

    return model
