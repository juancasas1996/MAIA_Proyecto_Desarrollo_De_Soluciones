import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn

df = pd.read_csv('features_eeg.csv')

X = df.drop(columns=['sujeto', 'edad']).values
y = df['edad'].values
grupos = df['sujeto'].values

logo = LeaveOneGroupOut()

mlflow.set_experiment('brain-age-index-local')

with mlflow.start_run(run_name='ridge-alpha100-normalizado-isabella'):
    maes = []
    r2s = []

    for train_idx, test_idx in logo.split(X, y, groups=grupos):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Normalizamos usando SOLO estadisticas del train (para no hacer trampa)
        escalador = StandardScaler()
        X_train_norm = escalador.fit_transform(X_train)
        X_test_norm = escalador.transform(X_test)

        modelo = Ridge(alpha=100)
        modelo.fit(X_train_norm, y_train)

        predicciones = modelo.predict(X_test_norm)
        maes.append(mean_absolute_error(y_test, predicciones))
        r2s.append(r2_score(y_test, predicciones))

    mae_promedio = np.mean(maes)
    r2_promedio = np.mean(r2s)

    print(f'MAE promedio (8 rondas): {mae_promedio:.2f} anios')
    print(f'R2 promedio (8 rondas): {r2_promedio:.2f}')

    mlflow.log_param('modelo', 'Ridge')
    mlflow.log_param('alpha', 100)
    mlflow.log_param('normalizado', True)
    mlflow.log_param('validacion', 'LeaveOneGroupOut')
    mlflow.log_metric('MAE', mae_promedio)
    mlflow.log_metric('R2', r2_promedio)

    # Entrenamos un modelo final con TODOS los datos para guardarlo
    escalador_final = StandardScaler()
    X_norm_final = escalador_final.fit_transform(X)
    modelo_final = Ridge(alpha=100)
    modelo_final.fit(X_norm_final, y)
    mlflow.sklearn.log_model(modelo_final, 'modelo')

print('Listo, revisa MLflow con: mlflow ui')