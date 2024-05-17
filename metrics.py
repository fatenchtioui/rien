# Définir la fonction pour calculer les métriques
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
def calculate_metrics(model, X, y_true):
    y_pred = model.predict(X)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mse = mean_squared_error(y_true, y_pred)
    return r2, rmse, mse