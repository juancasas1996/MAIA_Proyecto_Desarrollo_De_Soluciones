"""Random Forest: comparación de representaciones EEG.

Compara tres conjuntos de características:
A. Fpz-Cz
B. Pz-Oz
C. Fpz-Cz + Pz-Oz

Utiliza la partición oficial del proyecto:
- 62 sujetos para entrenamiento/validación.
- 16 sujetos de test.
- 5 folds predefinidos.

La selección del modelo se realiza exclusivamente con validación cruzada.
El test se utiliza una sola vez, sobre la mejor configuración global.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline


SEMILLA = 42

ESPECTRALES = (
    ["potencia_total_log"]
    + [f"abs_{b}" for b in ("delta", "theta", "alpha", "sigma", "beta")]
    + [f"rel_{b}" for b in ("delta", "theta", "alpha", "sigma", "beta")]
    + ["sef95"]
)

ARQUITECTURA = [
    "horas_dormidas",
    "eficiencia",
    "minutos_despierto",
    "minutos_hasta_rem",
    "pct_sueno_en_N1",
    "pct_sueno_en_N2",
    "pct_sueno_en_N3",
    "pct_sueno_en_REM",
]


def encontrar_raiz() -> Path:
    """Encuentra la raíz del repositorio Git."""
    inicio = Path(__file__).resolve().parent

    for ruta in [inicio, *inicio.parents]:
        if (ruta / ".git").exists():
            return ruta

    raise FileNotFoundError("No se encontró la raíz del repositorio.")


def cargar_canal(ruta: Path) -> pd.DataFrame:
    """Convierte las dos noches de cada sujeto en una sola fila."""
    df = pd.read_csv(ruta)

    columnas = (
        ["subject", "age"]
        + [f"mix_{f}" for f in ESPECTRALES]
        + ARQUITECTURA
    )

    faltantes = set(columnas) - set(df.columns)
    if faltantes:
        raise ValueError(
            f"Faltan columnas en {ruta.name}: {sorted(faltantes)}"
        )

    df = df[columnas].copy()

    agregacion = {
        columna: "mean"
        for columna in columnas
        if columna not in {"subject", "age"}
    }
    agregacion["age"] = "first"

    return df.groupby("subject").agg(agregacion).sort_index()


def metricas(y_real, y_pred) -> dict:
    """Calcula métricas de regresión."""
    return {
        "mae": float(mean_absolute_error(y_real, y_pred)),
        "rmse": float(
            np.sqrt(mean_squared_error(y_real, y_pred))
        ),
        "r2": float(r2_score(y_real, y_pred)),
    }


def crear_modelo(parametros: dict) -> Pipeline:
    """Pipeline de imputación + Random Forest."""
    return Pipeline(
        [
            ("imputar", SimpleImputer(strategy="median")),
            (
                "rf",
                RandomForestRegressor(
                    **parametros,
                    random_state=SEMILLA,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def evaluar_cv(
    X: pd.DataFrame,
    y: pd.Series,
    folds: list,
    parametros: dict,
) -> dict:
    """Evalúa una configuración usando los folds oficiales."""
    resultados = []

    for fold in folds:
        ids_train = fold["train_subject_ids"]
        ids_val = fold["val_subject_ids"]

        modelo = crear_modelo(parametros)

        modelo.fit(
            X.loc[ids_train],
            y.loc[ids_train],
        )

        pred = modelo.predict(X.loc[ids_val])

        m = metricas(
            y.loc[ids_val],
            pred,
        )

        resultados.append(m)

    return {
        "mae_cv": float(
            np.mean([x["mae"] for x in resultados])
        ),
        "mae_std": float(
            np.std([x["mae"] for x in resultados], ddof=1)
        ),
        "rmse_cv": float(
            np.mean([x["rmse"] for x in resultados])
        ),
        "r2_cv": float(
            np.mean([x["r2"] for x in resultados])
        ),
    }


def main() -> None:
    raiz = encontrar_raiz()

    ruta_fpz = (
        raiz
        / "Experimentos"
        / "Sebastian"
        / "datos"
        / "caracteristicas_noche_FpzCz.csv"
    )

    ruta_pz = (
        raiz
        / "Experimentos"
        / "Sebastian"
        / "datos"
        / "caracteristicas_noche_PzOz.csv"
    )

    ruta_split = (
        raiz
        / "Experimentos"
        / "subject_split_seed42.json"
    )

    salida = (
        raiz
        / "Modelos"
        / "JuanCamilo"
        / "resultados_features"
    )

    salida.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # Datos
    # ============================================================

    fpz = cargar_canal(ruta_fpz)
    pz = cargar_canal(ruta_pz)

    if not fpz.index.equals(pz.index):
        raise ValueError(
            "Los sujetos de los dos canales no coinciden."
        )

    if not np.allclose(fpz["age"], pz["age"]):
        raise ValueError(
            "Las edades no coinciden entre canales."
        )

    espec_fpz = fpz[
        [f"mix_{f}" for f in ESPECTRALES]
    ].copy()

    espec_fpz.columns = [
        f"{c}_FpzCz"
        for c in espec_fpz.columns
    ]

    espec_pz = pz[
        [f"mix_{f}" for f in ESPECTRALES]
    ].copy()

    espec_pz.columns = [
        f"{c}_PzOz"
        for c in espec_pz.columns
    ]

    arquitectura = fpz[ARQUITECTURA].copy()

    edad = fpz["age"].copy()

    datos = pd.concat(
        [
            espec_fpz,
            espec_pz,
            arquitectura,
            edad.rename("edad"),
        ],
        axis=1,
    )

    representaciones = {
        "A_FpzCz": (
            list(espec_fpz.columns)
            + ARQUITECTURA
        ),
        "B_PzOz": (
            list(espec_pz.columns)
            + ARQUITECTURA
        ),
        "C_AmbosCanales": (
            list(espec_fpz.columns)
            + list(espec_pz.columns)
            + ARQUITECTURA
        ),
    }

    print("\n=== DATOS ===")
    print(f"Sujetos: {len(datos)}")
    print(
        f"Edad: {datos.edad.min():.0f}"
        f" - {datos.edad.max():.0f}"
    )

    for nombre, columnas in representaciones.items():
        print(
            f"{nombre}: {len(columnas)} características"
        )

    # ============================================================
    # Partición oficial
    # ============================================================

    split = json.loads(
        ruta_split.read_text(encoding="utf-8")
    )

    train_validation = split[
        "train_validation_subject_ids"
    ]

    test = split["test_subject_ids"]

    folds = split["cv_folds"]

    print("\n=== PARTICIÓN ===")
    print(
        f"Train/validation: {len(train_validation)}"
    )
    print(f"Test: {len(test)}")
    print(f"Folds: {len(folds)}")

    # ============================================================
    # Grid de Random Forest
    # ============================================================

    grid = {
        "n_estimators": [200, 500],
        "max_depth": [None, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt"],
    }

    configuraciones = list(ParameterGrid(grid))

    print(
        f"\nConfiguraciones por representación: "
        f"{len(configuraciones)}"
    )

    resultados = []

    # ============================================================
    # Validación cruzada
    # ============================================================

    for nombre, columnas in representaciones.items():

        print(
            f"\n===== {nombre} "
            f"({len(columnas)} features) ====="
        )

        X = datos[columnas]
        y = datos["edad"]

        for i, parametros in enumerate(
            configuraciones,
            start=1,
        ):
            m = evaluar_cv(
                X,
                y,
                folds,
                parametros,
            )

            fila = {
                "representacion": nombre,
                "numero_features": len(columnas),
                **parametros,
                **m,
            }

            resultados.append(fila)

            print(
                f"[{i:02d}/{len(configuraciones)}] "
                f"MAE={m['mae_cv']:.3f} "
                f"RMSE={m['rmse_cv']:.3f} "
                f"R2={m['r2_cv']:.3f}"
            )

    tabla = (
        pd.DataFrame(resultados)
        .sort_values("mae_cv")
        .reset_index(drop=True)
    )

    tabla.to_csv(
        salida / "resultados_cv_features.csv",
        index=False,
    )

    # ============================================================
    # Mejor configuración global
    # ============================================================

    mejor = tabla.iloc[0]

    representacion_ganadora = mejor[
        "representacion"
    ]

    columnas_ganadoras = representaciones[
        representacion_ganadora
    ]

    mejores_parametros = {
        "n_estimators": int(
            mejor["n_estimators"]
        ),
        "max_depth": (
            None
            if pd.isna(mejor["max_depth"])
            else int(mejor["max_depth"])
        ),
        "min_samples_leaf": int(
            mejor["min_samples_leaf"]
        ),
        "max_features": str(
            mejor["max_features"]
        ),
    }

    print("\n=== GANADOR EN VALIDACIÓN ===")
    print(
        f"Representación: {representacion_ganadora}"
    )
    print(
        f"Features: {len(columnas_ganadoras)}"
    )
    print(
        f"MAE CV: {mejor['mae_cv']:.3f}"
    )
    print(
        f"RMSE CV: {mejor['rmse_cv']:.3f}"
    )
    print(
        f"R2 CV: {mejor['r2_cv']:.3f}"
    )
    print(
        f"Parámetros: {mejores_parametros}"
    )

    # ============================================================
    # TEST: UNA SOLA VEZ
    # ============================================================

    X = datos[columnas_ganadoras]
    y = datos["edad"]

    modelo_final = crear_modelo(
        mejores_parametros
    )

    modelo_final.fit(
        X.loc[train_validation],
        y.loc[train_validation],
    )

    pred_test = modelo_final.predict(
        X.loc[test]
    )

    metricas_test = metricas(
        y.loc[test],
        pred_test,
    )

    print("\n=== TEST FINAL ===")
    print(
        f"MAE: {metricas_test['mae']:.3f}"
    )
    print(
        f"RMSE: {metricas_test['rmse']:.3f}"
    )
    print(
        f"R2: {metricas_test['r2']:.3f}"
    )

    # ============================================================
    # Predicciones
    # ============================================================

    predicciones = pd.DataFrame(
        {
            "subject": test,
            "edad_real": y.loc[test].values,
            "edad_estimada": pred_test,
        }
    )

    predicciones["error_absoluto"] = np.abs(
        predicciones["edad_real"]
        - predicciones["edad_estimada"]
    )

    predicciones.to_csv(
        salida / "predicciones_test.csv",
        index=False,
    )

    # ============================================================
    # Importancias
    # ============================================================

    rf = modelo_final.named_steps["rf"]

    importancias = (
        pd.DataFrame(
            {
                "caracteristica": columnas_ganadoras,
                "importancia": rf.feature_importances_,
            }
        )
        .sort_values(
            "importancia",
            ascending=False,
        )
    )

    importancias.to_csv(
        salida / "importancia_features.csv",
        index=False,
    )

    # ============================================================
    # Resumen
    # ============================================================

    resumen = {
        "representacion_ganadora":
            representacion_ganadora,
        "numero_features":
            len(columnas_ganadoras),
        "mejores_parametros":
            mejores_parametros,
        "metricas_validacion": {
            "mae": float(mejor["mae_cv"]),
            "rmse": float(mejor["rmse_cv"]),
            "r2": float(mejor["r2_cv"]),
        },
        "metricas_test":
            metricas_test,
    }

    (
        salida / "resumen_features.json"
    ).write_text(
        json.dumps(
            resumen,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        f"\nResultados guardados en:\n{salida}"
    )

    print("\n=== MEJORES POR REPRESENTACIÓN ===")

    mejores_representacion = (
        tabla
        .sort_values("mae_cv")
        .groupby("representacion")
        .first()
        .reset_index()
    )

    print(
        mejores_representacion[
            [
                "representacion",
                "numero_features",
                "mae_cv",
                "rmse_cv",
                "r2_cv",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
