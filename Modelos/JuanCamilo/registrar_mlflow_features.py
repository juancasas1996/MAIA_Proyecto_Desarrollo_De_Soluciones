"""Registra en MLflow los resultados del experimento Random Forest por representación."""

from pathlib import Path
import json

import mlflow
import pandas as pd


EXPERIMENTO = "edad-cerebral-JuanCamilo-RandomForest-Features"


def encontrar_raiz() -> Path:
    inicio = Path(__file__).resolve().parent
    for ruta in [inicio, *inicio.parents]:
        if (ruta / ".git").exists():
            return ruta
    raise FileNotFoundError("No se encontró la raíz del repositorio.")


def main():
    raiz = encontrar_raiz()

    resultados_dir = (
        raiz
        / "Modelos"
        / "JuanCamilo"
        / "resultados_features"
    )

    csv_resultados = resultados_dir / "resultados_cv_features.csv"
    csv_predicciones = resultados_dir / "predicciones_test.csv"
    csv_importancias = resultados_dir / "importancia_features.csv"
    json_resumen = resultados_dir / "resumen_features.json"
    split = raiz / "Experimentos" / "subject_split_seed42.json"

    # Importar el conector común del proyecto
    import sys
    sys.path.insert(0, str(raiz / "Experimentos"))

    from config import conectar

    uri = conectar()
    mlflow.set_experiment(EXPERIMENTO)

    print(f"Registrando en: {uri}")
    print(f"Experimento: {EXPERIMENTO}")

    tabla = pd.read_csv(csv_resultados)

    # ==========================================================
    # 1. Registrar las 54 configuraciones
    # ==========================================================

    for representacion, grupo in tabla.groupby("representacion"):

        mejor = grupo.sort_values("mae_cv").iloc[0]

        with mlflow.start_run(
            run_name=f"{representacion}_resumen"
        ) as run_padre:

            mlflow.set_tags(
                {
                    "autor": "Juan Camilo Martinez Velez",
                    "modelo": "RandomForestRegressor",
                    "tipo_run": "representacion",
                    "representacion": representacion,
                    "protocolo": "split_oficial_62_16",
                }
            )

            mlflow.log_param(
                "numero_features",
                int(mejor["numero_features"]),
            )

            mlflow.log_metrics(
                {
                    "mejor_mae_cv": float(mejor["mae_cv"]),
                    "mejor_rmse_cv": float(mejor["rmse_cv"]),
                    "mejor_r2_cv": float(mejor["r2_cv"]),
                }
            )

            for numero, fila in enumerate(
                grupo.itertuples(index=False),
                start=1,
            ):
                with mlflow.start_run(
                    run_name=f"{representacion}_rf_{numero:02d}",
                    nested=True,
                ):
                    mlflow.set_tags(
                        {
                            "autor": "Juan Camilo Martinez Velez",
                            "modelo": "RandomForestRegressor",
                            "tipo_run": "rejilla",
                            "representacion": representacion,
                        }
                    )

                    mlflow.log_params(
                        {
                            "numero_features":
                                int(fila.numero_features),
                            "n_estimators":
                                int(fila.n_estimators),
                            "max_depth":
                                "None"
                                if pd.isna(fila.max_depth)
                                else int(fila.max_depth),
                            "min_samples_leaf":
                                int(fila.min_samples_leaf),
                            "max_features":
                                str(fila.max_features),
                            "random_state": 42,
                        }
                    )

                    mlflow.log_metrics(
                        {
                            "mae_cv": float(fila.mae_cv),
                            "mae_std": float(fila.mae_std),
                            "rmse_cv": float(fila.rmse_cv),
                            "r2_cv": float(fila.r2_cv),
                        }
                    )

    # ==========================================================
    # 2. Registrar el ganador global
    # ==========================================================

    resumen = json.loads(
        json_resumen.read_text(encoding="utf-8")
    )

    with mlflow.start_run(
        run_name="RandomForest_Features_Ganador"
    ):

        mlflow.set_tags(
            {
                "autor": "Juan Camilo Martinez Velez",
                "modelo": "RandomForestRegressor",
                "tipo_run": "ganador_global",
                "representacion":
                    resumen["representacion_ganadora"],
                "protocolo": "split_oficial_62_16",
            }
        )

        mlflow.log_params(
            {
                "representacion_ganadora":
                    resumen["representacion_ganadora"],
                "numero_features":
                    resumen["numero_features"],
                **{
                    f"rf_{k}": (
                        "None" if v is None else v
                    )
                    for k, v
                    in resumen["mejores_parametros"].items()
                },
                "train_validation_subjects": 62,
                "test_subjects": 16,
                "cv_folds": 5,
            }
        )

        validacion = resumen["metricas_validacion"]
        test = resumen["metricas_test"]

        mlflow.log_metrics(
            {
                "mae_cv": validacion["mae"],
                "rmse_cv": validacion["rmse"],
                "r2_cv": validacion["r2"],
                "mae_test": test["mae"],
                "rmse_test": test["rmse"],
                "r2_test": test["r2"],
            }
        )

        # Artefactos reproducibles
        for archivo in [
            csv_resultados,
            csv_predicciones,
            csv_importancias,
            json_resumen,
            split,
        ]:
            mlflow.log_artifact(
                str(archivo),
                artifact_path="resultados",
            )

    print("\nRegistro completado correctamente.")
    print(
        "Ganador:",
        resumen["representacion_ganadora"],
    )
    print(
        "MAE CV:",
        resumen["metricas_validacion"]["mae"],
    )
    print(
        "MAE test:",
        resumen["metricas_test"]["mae"],
    )


if __name__ == "__main__":
    main()
