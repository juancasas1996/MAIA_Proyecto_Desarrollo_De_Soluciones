import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn


def encontrar_raiz_repositorio():
    inicio_script = Path(__file__).resolve().parent
    candidatos = [Path.cwd(), *Path.cwd().parents, inicio_script, *inicio_script.parents]
    for ruta in candidatos:
        if (ruta / '.git').exists():
            return ruta.resolve()
    raise FileNotFoundError('No se encontro la raiz del repositorio Git.')


# ============================================
# Cargar datos y particion
# ============================================
raiz = encontrar_raiz_repositorio()
datos_sebastian = raiz / 'Experimentos' / 'Sebastian' / 'datos'
df_fpzcz = pd.read_csv(datos_sebastian / 'caracteristicas_noche_FpzCz.csv')
df_pzoz = pd.read_csv(datos_sebastian / 'caracteristicas_noche_PzOz.csv')

with (raiz / 'Experimentos' / 'subject_split_seed42.json').open(encoding='utf-8') as f:
    particion = json.load(f)

test_ids = set(particion['test_subject_ids'])
dev_ids = set(particion['train_validation_subject_ids'])
folds = particion['cv_folds']

# ============================================
# IMPORTANTE: la unidad es el SUJETO, no la noche
# Promediamos las dos noches de cada sujeto (regla del proyecto)
# ============================================
def promediar_por_sujeto(df):
    # Seleccionamos SOLO columnas numericas (esto excluye automaticamente
    # texto como 'sex' o 'registro', que no se pueden promediar)
    columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'subject' in columnas_numericas:
        columnas_numericas.remove('subject')
    return df.groupby('subject')[columnas_numericas].mean().reset_index()


# Usamos el canal Fpz-Cz.
df_sujeto = promediar_por_sujeto(df_fpzcz)

X_cols = [c for c in df_sujeto.columns if c not in ['subject', 'age']]
X = df_sujeto[X_cols].values
y = df_sujeto['age'].values
sujetos = df_sujeto['subject'].values

def filtrar(ids_deseados):
    mask = np.isin(sujetos, list(ids_deseados))
    return X[mask], y[mask]

# ============================================
# Validacion cruzada (los 5 folds que ya vienen definidos)
# ============================================
mlflow.set_experiment('edad-cerebral-isabella-rf')

with mlflow.start_run(run_name='random-forest-FpzCz-isabella'):
    maes_val = []
    r2s_val = []

    for fold_info in folds:
        train_ids = fold_info['train_subject_ids']
        val_ids = fold_info['val_subject_ids']

        X_train, y_train = filtrar(train_ids)
        X_val, y_val = filtrar(val_ids)

        modelo = RandomForestRegressor(n_estimators=300, max_depth=5, random_state=42)
        modelo.fit(X_train, y_train)

        pred_val = modelo.predict(X_val)
        maes_val.append(mean_absolute_error(y_val, pred_val))
        r2s_val.append(r2_score(y_val, pred_val))

    mae_val_promedio = np.mean(maes_val)
    r2_val_promedio = np.mean(r2s_val)

    print(f'MAE validacion cruzada (5 folds): {mae_val_promedio:.2f} anios')
    print(f'R2 validacion cruzada (5 folds): {r2_val_promedio:.2f}')

    # ============================================
    # Entrenamos con TODO development, evaluamos en TEST (una sola vez)
    # ============================================
    X_dev, y_dev = filtrar(dev_ids)
    X_test, y_test = filtrar(test_ids)

    modelo_final = RandomForestRegressor(n_estimators=300, max_depth=5, random_state=42)
    modelo_final.fit(X_dev, y_dev)

    pred_train = modelo_final.predict(X_dev)
    pred_test = modelo_final.predict(X_test)

    mae_train = mean_absolute_error(y_dev, pred_train)
    mae_test = mean_absolute_error(y_test, pred_test)
    r2_test = r2_score(y_test, pred_test)

    # Referencia: predecir siempre la edad media (igual que hizo Sebastian)
    media_edad = y_dev.mean()
    mae_referencia = mean_absolute_error(y_test, [media_edad]*len(y_test))

    print(f'MAE train (todo development): {mae_train:.2f} anios')
    print(f'MAE test (16 sujetos, mirado UNA vez): {mae_test:.2f} anios')
    print(f'R2 test: {r2_test:.2f}')
    print(f'Referencia (predecir la media): {mae_referencia:.2f} anios')

    mlflow.log_param('modelo', 'RandomForest')
    mlflow.log_param('n_estimators', 300)
    mlflow.log_param('max_depth', 5)
    mlflow.log_param('canal', 'Fpz-Cz')
    mlflow.log_param('particion', 'subject_split_seed42 (misma que Sebastian)')
    mlflow.log_metric('MAE_train', mae_train)
    mlflow.log_metric('MAE_val_cv', mae_val_promedio)
    mlflow.log_metric('R2_val_cv', r2_val_promedio)
    mlflow.log_metric('MAE_test', mae_test)
    mlflow.log_metric('R2_test', r2_test)
    mlflow.log_metric('MAE_referencia_media', mae_referencia)
    mlflow.sklearn.log_model(modelo_final, 'modelo')

print('Listo, revisa MLflow con: mlflow ui')
