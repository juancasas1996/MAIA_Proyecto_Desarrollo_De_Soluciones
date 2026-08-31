# Exploracion de Random Forest para estimar la edad

## Objetivo

Evaluar si un modelo `RandomForestRegressor` puede mejorar la linea base Ridge
para estimar la edad cronologica a partir de caracteristicas resumidas del EEG
del sueno.

## Datos

El experimento utiliza `Modelos/datos/features_por_sujeto.csv`, generado a
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
python Modelos/JuanCamilo/experimento_random_forest.py
```

Si no se define `MLFLOW_TRACKING_URI`, el script crea automáticamente una base
SQLite local en `mlruns/mlflow.db`. Para registrar los experimentos en un
servidor remoto:

```bash
export MLFLOW_TRACKING_URI="http://IP_DEL_SERVIDOR:5000"
python Modelos/JuanCamilo/experimento_random_forest.py
```

Para consultar los experimentos locales:

```bash
mlflow ui --backend-store-uri "sqlite:///$(pwd)/mlruns/mlflow.db" --host 127.0.0.1 --port 5000
```

Luego se abre `http://127.0.0.1:5000` y se selecciona el experimento
`SomnoAI-JuanCamilo-RandomForest` en la seccion **Model training**.

## Archivos generados

El directorio `Modelos/JuanCamilo/resultados` contiene:

- `resultados_random_forest.csv`: las 18 configuraciones evaluadas.
- `comparacion_modelos.csv`: comparacion entre media, Ridge y Random Forest.
- `predicciones_random_forest.csv`: predicciones fuera de muestra por sujeto.
- `importancia_caracteristicas.csv`: importancias del modelo final.
- `edad_real_vs_estimada.png`: grafica de predicciones.
- `importancia_caracteristicas.png`: principales variables.
- `resumen_experimento.json`: parametros, metricas y configuracion general.

---

# Experimento 2 — Random Forest con diferentes representaciones EEG

## Objetivo

Evaluar si el desempeño de Random Forest para estimar la edad cronológica mejora al utilizar diferentes representaciones de las señales EEG, manteniendo la partición y el protocolo de evaluación común del proyecto.

Se compararon tres representaciones:

| Representación | Características |
|---|---:|
| A — Fpz-Cz | 20 |
| B — Pz-Oz | 20 |
| C — Fpz-Cz + Pz-Oz | 32 |

Las variables combinan características espectrales del EEG con información de arquitectura del sueño.

## Protocolo de evaluación

Se utilizó la partición común definida en `Experimentos/subject_split_seed42.json`:

- 78 sujetos en total.
- 62 sujetos para entrenamiento y validación.
- 16 sujetos reservados para test.
- 5 folds predefinidos de validación cruzada.
- Semilla 42.

La representación y los hiperparámetros se seleccionaron exclusivamente mediante validación cruzada sobre los 62 sujetos de desarrollo. El conjunto de test se utilizó una sola vez al final sobre la configuración seleccionada.

## Búsqueda de hiperparámetros

Para cada representación se evaluaron 18 configuraciones de `RandomForestRegressor`:

- `n_estimators`: 200 y 500.
- `max_depth`: sin límite, 5 y 10.
- `min_samples_leaf`: 1, 2 y 4.
- `max_features`: `sqrt`.

En total se realizaron 54 evaluaciones.

## Resultados de validación

| Representación | Features | MAE CV | RMSE CV | R² CV |
|---|---:|---:|---:|---:|
| A — Fpz-Cz | 20 | 13.869 | 15.345 | 0.466 |
| **B — Pz-Oz** | **20** | **13.279** | **14.999** | **0.484** |
| C — Ambos canales | 32 | 13.826 | 15.267 | 0.471 |

La mejor representación fue Pz-Oz con 20 características.

La mejor configuración fue:

- `n_estimators = 200`
- `max_depth = None`
- `min_samples_leaf = 1`
- `max_features = sqrt`

## Evaluación final en test

El modelo seleccionado se entrenó con los 62 sujetos de desarrollo y se evaluó posteriormente sobre los 16 sujetos reservados para test.

| Métrica | Resultado |
|---|---:|
| MAE | 13.762 años |
| RMSE | 16.784 años |
| R² | 0.448 |

El uso simultáneo de los dos canales EEG no mejoró el desempeño frente al uso exclusivo de Pz-Oz. Para el tamaño de muestra disponible, aumentar el número de características de 20 a 32 no incrementó la capacidad de generalización del Random Forest.

## Características más relevantes

Las variables con mayor importancia en el modelo ganador fueron:

1. `pct_sueno_en_N1`
2. `mix_rel_beta_PzOz`
3. `eficiencia`
4. `minutos_despierto`
5. `mix_rel_delta_PzOz`
6. `pct_sueno_en_REM`
7. `mix_sef95_PzOz`
8. `mix_abs_delta_PzOz`
9. `mix_rel_alpha_PzOz`
10. `mix_potencia_total_log_PzOz`

El modelo utiliza tanto información sobre arquitectura del sueño como características espectrales del canal Pz-Oz. Estas importancias representan contribución predictiva dentro del Random Forest y no relaciones causales.

## Seguimiento con MLflow

Los resultados se registraron en el servidor MLflow compartido bajo el experimento `edad-cerebral-JuanCamilo-RandomForest-Features`.

Se registraron:

- 18 configuraciones para Fpz-Cz.
- 18 configuraciones para Pz-Oz.
- 18 configuraciones para ambos canales.
- Una corrida resumen para cada representación.
- Una corrida final `RandomForest_Features_Ganador`.

En total se registraron 58 corridas.

## Ejecución

El experimento se ejecuta desde la raíz del repositorio con:

    python Modelos/JuanCamilo/experimento_random_forest_features.py

Para registrar los resultados previamente generados en el MLflow compartido:

    export MLFLOW_TRACKING_URI="http://IP_MLFLOW:8050"
    python Modelos/JuanCamilo/registrar_mlflow_features.py

La IP del servidor no se almacena permanentemente porque la dirección pública de la instancia EC2 puede cambiar entre sesiones.

## Archivos generados

El directorio `Modelos/JuanCamilo/resultados_features/` contiene:

- `resultados_cv_features.csv`: resultados de las 54 configuraciones.
- `predicciones_test.csv`: predicciones del modelo ganador sobre los 16 sujetos de test.
- `importancia_features.csv`: importancia de variables del modelo ganador.
- `resumen_features.json`: configuración y métricas finales del experimento.

