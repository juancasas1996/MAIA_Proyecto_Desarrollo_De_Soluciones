# Exploracion de Random Forest para estimar la edad

## Objetivo

Evaluar si un modelo `RandomForestRegressor` puede mejorar la linea base Ridge
para estimar la edad cronologica a partir de caracteristicas resumidas del EEG
del sueno.

## Datos

El experimento utiliza `Experimentos/JuanCamilo/datos/features_por_sujeto.csv`, generado a
partir del DataFrame `datos` de `EDA/EDA_Consolidado.ipynb`.

- 78 sujetos.
- 20 caracteristicas predictoras.
- Variable objetivo: `age`.
- Identificador de agrupacion: `subject`.
- Columnas excluidas del entrenamiento: `subject`, `age` y `sex`.
- Sin valores nulos.

## Metodologia

Se evaluaron 18 configuraciones de Random Forest mediante validacion cruzada
agrupada de cinco particiones (`GroupKFold`) y semilla 42. La rejilla exploro:

- `n_estimators`: 200 y 500.
- `max_depth`: sin limite, 5 y 10.
- `min_samples_leaf`: 1, 2 y 4.
- `max_features`: `sqrt`.

La metrica principal fue MAE, expresada directamente en anos. Tambien se
calcularon RMSE y R2. Cada configuracion se registro como una ejecucion en
MLflow, junto con una ejecucion resumen del mejor modelo.

## Resultados

| Modelo | MAE | RMSE | R2 |
|---|---:|---:|---:|
| Ridge | 10.20 | 13.17 | 0.642 |
| Random Forest | 13.15 | 15.60 | 0.498 |
| Prediccion de la media | 18.41 | 22.18 | -0.016 |

La mejor configuracion de Random Forest fue:

- `n_estimators = 500`
- `max_depth = 5`
- `min_samples_leaf = 1`
- `max_features = sqrt`

Random Forest aprendio informacion relevante y redujo el MAE de la prediccion
de la media en aproximadamente 5.26 anos. Sin embargo, no supero a Ridge, cuyo
MAE fue aproximadamente 2.95 anos menor. Con una muestra de 78 sujetos, la
regularizacion lineal de Ridge presento una mejor capacidad de generalizacion.

La grafica de edad real frente a edad estimada muestra regresion hacia la
media: el modelo tiende a sobreestimar a los sujetos jovenes y a subestimar a
los de mayor edad.

Las caracteristicas con mayor importancia predictiva fueron `pct_N1`,
`eficiencia`, `waso_min`, `rel_beta` y `abs_sigma`. Estas importancias no deben
interpretarse como relaciones causales.

## Ejecucion

Desde la raiz del repositorio, con el ambiente virtual activo:

```bash
python Experimentos/JuanCamilo/experimento_random_forest.py
```

Si no se define `MLFLOW_TRACKING_URI`, el script crea automáticamente una base
SQLite local en `mlruns/mlflow.db`. Para registrar los experimentos en un
servidor remoto:

```bash
export MLFLOW_TRACKING_URI="http://IP_DEL_SERVIDOR:5000"
python Experimentos/JuanCamilo/experimento_random_forest.py
```

Para consultar los experimentos locales:

```bash
mlflow ui --backend-store-uri "sqlite:///$(pwd)/mlruns/mlflow.db" --host 127.0.0.1 --port 5000
```

Luego se abre `http://127.0.0.1:5000` y se selecciona el experimento
`SomnoAI-JuanCamilo-RandomForest` en la seccion **Model training**.

## Archivos generados

El directorio `Experimentos/JuanCamilo/resultados` contiene:

- `resultados_random_forest.csv`: las 18 configuraciones evaluadas.
- `comparacion_modelos.csv`: comparacion entre media, Ridge y Random Forest.
- `predicciones_random_forest.csv`: predicciones fuera de muestra por sujeto.
- `importancia_caracteristicas.csv`: importancias del modelo final.
- `edad_real_vs_estimada.png`: grafica de predicciones.
- `importancia_caracteristicas.png`: principales variables.
- `resumen_experimento.json`: parametros, metricas y configuracion general.
