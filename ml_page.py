

import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
import seaborn as sns
import matplotlib.pyplot as plt
from colorama import Fore
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def get_random_forest_model(train_data, feature_columns, target_column):


    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)

    model_rf.fit(train_data[feature_columns], train_data[target_column])

    return model_rf




def get_br(train_data, feature_columns, target_column):

    # Define BaggingRegressor model
    model_bagging = BaggingRegressor(n_estimators=100, random_state=42)
    
   
    model_bagging.fit(train_data[feature_columns], train_data[target_column])
    return model_bagging
def get_gb(train_data, feature_columns, target_column):
    # Define GradientBoostingRegressor model
    model_gradient_boosting = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model_gradient_boosting.fit(train_data[feature_columns], train_data[target_column])
    return model_gradient_boosting



