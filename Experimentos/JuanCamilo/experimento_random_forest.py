"""Exploracion de Random Forest para estimar edad a partir del EEG del sueno.

El script compara 18 configuraciones mediante validacion cruzada agrupada,
registra cada experimento en MLflow y genera tablas y graficas reproducibles.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GroupKFold,
    ParameterGrid,
    cross_val_predict,
    cross_validate,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


EXPERIMENTO = "SomnoAI-JuanCamilo-RandomForest"
SEMILLA = 42
PLIEGUES = 5


def encontrar_raiz_repositorio() -> Path:
    """Encuentra la raiz del repositorio independientemente del directorio actual."""
    inicio_script = Path(__file__).resolve().parent
    candidatos = [Path.cwd(), *Path.cwd().parents, inicio_script, *inicio_script.parents]
    for ruta in candidatos:
        if (ruta / ".git").exists():
            return ruta.resolve()
    raise FileNotFoundError("No se encontro la raiz del repositorio Git.")


def convertir_parametros_mlflow(parametros: dict) -> dict:
    """Convierte valores None a texto para registrarlos de forma estable."""
    return {
        clave: "None" if valor is None else valor
        for clave, valor in parametros.items()
    }


def calcular_metricas(y_real: pd.Series, y_estimada: np.ndarray) -> dict:
    """Calcula las metricas globales de las predicciones fuera de muestra."""
    return {
        "mae": float(mean_absolute_error(y_real, y_estimada)),
        "rmse": float(np.sqrt(mean_squared_error(y_real, y_estimada))),
        "r2": float(r2_score(y_real, y_estimada)),
    }


def guardar_grafica_predicciones(
    y_real: pd.Series,
    y_estimada: np.ndarray,
    metricas: dict,
    destino: Path,
) -> None:
    """Guarda la grafica de edad real frente a edad estimada."""
    minimo = float(min(y_real.min(), y_estimada.min()))
    maximo = float(max(y_real.max(), y_estimada.max()))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_real, y_estimada, alpha=0.8, color="#2F5597")
    ax.plot([minimo, maximo], [minimo, maximo], "--", color="#C00000")
    ax.set_xlabel("Edad real (anos)")
    ax.set_ylabel("Edad estimada (anos)")
    ax.set_title("Random Forest: edad real frente a edad estimada")
    ax.text(
        0.03,
        0.97,
        f"MAE = {metricas['mae']:.2f}\nRMSE = {metricas['rmse']:.2f}\nR2 = {metricas['r2']:.3f}",
        transform=ax.transAxes,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    fig.tight_layout()
    fig.savefig(destino, dpi=180, bbox_inches="tight")
    plt.close(fig)


def guardar_grafica_importancias(importancias: pd.DataFrame, destino: Path) -> None:
    """Guarda las 15 caracteristicas mas importantes del modelo final."""
    principales = importancias.head(15).sort_values("importancia")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(principales["caracteristica"], principales["importancia"], color="#70AD47")
    ax.set_xlabel("Importancia")
    ax.set_title("Random Forest: importancia de caracteristicas")
    fig.tight_layout()
    fig.savefig(destino, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    raiz = encontrar_raiz_repositorio()
    archivo_datos = (
        raiz / "Experimentos" / "JuanCamilo" / "datos" / "features_por_sujeto.csv"
    )
    salida = raiz / "Experimentos" / "JuanCamilo" / "resultados"
    salida.mkdir(parents=True, exist_ok=True)

    if not archivo_datos.exists():
        raise FileNotFoundError(f"No se encontro el archivo: {archivo_datos}")

    datos = pd.read_csv(archivo_datos)
    requeridas = {"subject", "age", "sex"}
    faltantes = requeridas.difference(datos.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas obligatorias: {sorted(faltantes)}")

    caracteristicas = [
        columna
        for columna in datos.columns
        if columna not in {"subject", "age", "sex"}
    ]
    X = datos[caracteristicas]
    y = datos["age"]
    grupos = datos["subject"]

    if X.isna().any().any() or y.isna().any():
        raise ValueError("Los datos contienen valores nulos.")
    if len(caracteristicas) != 20:
        raise ValueError(
            f"Se esperaban 20 caracteristicas, pero se encontraron {len(caracteristicas)}."
        )

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        base_mlflow = raiz / "mlruns"
        base_mlflow.mkdir(parents=True, exist_ok=True)
        tracking_uri = f"sqlite:///{(base_mlflow / 'mlflow.db').resolve()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENTO)

    cv = GroupKFold(n_splits=PLIEGUES, shuffle=True, random_state=SEMILLA)
    rejilla = {
        "n_estimators": [200, 500],
        "max_depth": [None, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt"],
    }

    resultados: list[dict] = []
    configuraciones = list(ParameterGrid(rejilla))

    print(f"Datos: {datos.shape[0]} sujetos y {len(caracteristicas)} caracteristicas")
    print(f"MLflow: {tracking_uri}")
    print(f"Configuraciones por evaluar: {len(configuraciones)}\n")

    for numero, parametros in enumerate(configuraciones, start=1):
        modelo = RandomForestRegressor(
            **parametros,
            random_state=SEMILLA,
            n_jobs=-1,
        )
        evaluacion = cross_validate(
            modelo,
            X,
            y,
            groups=grupos,
            cv=cv,
            scoring={
                "mae": "neg_mean_absolute_error",
                "rmse": "neg_root_mean_squared_error",
                "r2": "r2",
            },
            n_jobs=1,
        )

        mae_pliegues = -evaluacion["test_mae"]
        rmse_pliegues = -evaluacion["test_rmse"]
        r2_pliegues = evaluacion["test_r2"]
        fila = {
            **parametros,
            "mae_cv": float(mae_pliegues.mean()),
            "mae_std": float(mae_pliegues.std(ddof=1)),
            "rmse_cv": float(rmse_pliegues.mean()),
            "r2_cv": float(r2_pliegues.mean()),
        }
        resultados.append(fila)

        with mlflow.start_run(run_name=f"rf_{numero:02d}"):
            mlflow.set_tags(
                {
                    "autor": "Juan Camilo Martinez Velez",
                    "modelo": "RandomForestRegressor",
                    "fuente_datos": "features_por_sujeto.csv",
                    "tipo_validacion": "GroupKFold",
                }
            )
            mlflow.log_params(convertir_parametros_mlflow(parametros))
            mlflow.log_param("cv_splits", PLIEGUES)
            mlflow.log_param("random_state", SEMILLA)
            mlflow.log_param("numero_sujetos", len(datos))
            mlflow.log_param("numero_caracteristicas", len(caracteristicas))
            mlflow.log_metrics(
                {
                    "mae_cv": fila["mae_cv"],
                    "mae_std": fila["mae_std"],
                    "rmse_cv": fila["rmse_cv"],
                    "r2_cv": fila["r2_cv"],
                }
            )

        print(
            f"[{numero:02d}/{len(configuraciones)}] "
            f"MAE={fila['mae_cv']:.3f} | RMSE={fila['rmse_cv']:.3f} | "
            f"R2={fila['r2_cv']:.3f} | {parametros}"
        )

    tabla_resultados = pd.DataFrame(resultados).sort_values("mae_cv").reset_index(drop=True)
    archivo_resultados = salida / "resultados_random_forest.csv"
    tabla_resultados.to_csv(archivo_resultados, index=False)

    mejor_fila = tabla_resultados.iloc[0]
    mejores_parametros = {
        "n_estimators": int(mejor_fila["n_estimators"]),
        "max_depth": None if pd.isna(mejor_fila["max_depth"]) else int(mejor_fila["max_depth"]),
        "min_samples_leaf": int(mejor_fila["min_samples_leaf"]),
        "max_features": str(mejor_fila["max_features"]),
    }
    mejor_modelo = RandomForestRegressor(
        **mejores_parametros,
        random_state=SEMILLA,
        n_jobs=-1,
    )

    predicciones_rf = cross_val_predict(
        mejor_modelo,
        X,
        y,
        groups=grupos,
        cv=cv,
        n_jobs=1,
    )
    metricas_rf = calcular_metricas(y, predicciones_rf)

    modelo_ridge = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=np.logspace(-3, 3, 50)),
    )
    predicciones_ridge = cross_val_predict(
        modelo_ridge,
        X,
        y,
        groups=grupos,
        cv=cv,
    )
    metricas_ridge = calcular_metricas(y, predicciones_ridge)

    modelo_media = DummyRegressor(strategy="mean")
    predicciones_media = cross_val_predict(
        modelo_media,
        X,
        y,
        groups=grupos,
        cv=cv,
    )
    metricas_media = calcular_metricas(y, predicciones_media)

    comparacion = pd.DataFrame(
        [
            {"modelo": "Media", **metricas_media},
            {"modelo": "Ridge", **metricas_ridge},
            {"modelo": "Random Forest", **metricas_rf},
        ]
    ).sort_values("mae")
    archivo_comparacion = salida / "comparacion_modelos.csv"
    comparacion.to_csv(archivo_comparacion, index=False)

    predicciones = datos[["subject", "age"]].copy()
    predicciones["edad_estimada_random_forest"] = predicciones_rf
    predicciones["error_absoluto_random_forest"] = np.abs(y.to_numpy() - predicciones_rf)
    archivo_predicciones = salida / "predicciones_random_forest.csv"
    predicciones.to_csv(archivo_predicciones, index=False)

    mejor_modelo.fit(X, y)
    importancias = pd.DataFrame(
        {
            "caracteristica": caracteristicas,
            "importancia": mejor_modelo.feature_importances_,
        }
    ).sort_values("importancia", ascending=False)
    archivo_importancias = salida / "importancia_caracteristicas.csv"
    importancias.to_csv(archivo_importancias, index=False)

    grafica_predicciones = salida / "edad_real_vs_estimada.png"
    grafica_importancias = salida / "importancia_caracteristicas.png"
    guardar_grafica_predicciones(y, predicciones_rf, metricas_rf, grafica_predicciones)
    guardar_grafica_importancias(importancias, grafica_importancias)

    resumen = {
        "experimento": EXPERIMENTO,
        "tracking_uri": tracking_uri,
        "numero_sujetos": len(datos),
        "numero_caracteristicas": len(caracteristicas),
        "numero_configuraciones": len(configuraciones),
        "mejores_parametros": mejores_parametros,
        "metricas_random_forest_oof": metricas_rf,
        "metricas_ridge_oof": metricas_ridge,
        "metricas_media_oof": metricas_media,
    }
    archivo_resumen = salida / "resumen_experimento.json"
    archivo_resumen.write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with mlflow.start_run(run_name="rf_mejor_modelo_resumen"):
        mlflow.set_tags(
            {
                "autor": "Juan Camilo Martinez Velez",
                "modelo": "RandomForestRegressor",
                "tipo_run": "resumen_mejor_modelo",
            }
        )
        mlflow.log_params(convertir_parametros_mlflow(mejores_parametros))
        mlflow.log_metrics(
            {
                "mae_oof": metricas_rf["mae"],
                "rmse_oof": metricas_rf["rmse"],
                "r2_oof": metricas_rf["r2"],
                "mae_ridge_oof": metricas_ridge["mae"],
                "mae_media_oof": metricas_media["mae"],
            }
        )
        for artefacto in [
            archivo_resultados,
            archivo_comparacion,
            archivo_predicciones,
            archivo_importancias,
            grafica_predicciones,
            grafica_importancias,
            archivo_resumen,
        ]:
            mlflow.log_artifact(str(artefacto), artifact_path="resultados")

    print("\nExperimento terminado correctamente.")
    print("\nCinco mejores configuraciones:")
    print(tabla_resultados.head(5).to_string(index=False))
    print("\nComparacion de modelos:")
    print(comparacion.to_string(index=False))
    print(f"\nResultados guardados en: {salida}")
    print(f"Experimento MLflow: {EXPERIMENTO}")


if __name__ == "__main__":
    main()
