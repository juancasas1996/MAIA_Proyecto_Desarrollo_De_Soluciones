"""Conexion con el servidor de MLflow del EC2.

Reemplaza al `export MLFLOW_TRACKING_URI=...`: la direccion vive en config.yaml
y no en la memoria de una terminal concreta, asi que sirve igual desde un script,
un notebook o una terminal recien abierta.

El nombre del experimento no vive aqui: lo elige cada notebook, que es quien
sabe que esta probando.

Uso:
    from config import conectar
    conectar()

    mlflow.set_experiment("arquitectura-vs-espectral")
    with mlflow.start_run(run_name="ridge-base"):
        ...
"""

import os
import urllib.request
from pathlib import Path

import mlflow
import yaml

RUTA = Path(__file__).parent / "config.yaml"


def cargar() -> dict:
    """Devuelve el contenido de config.yaml."""
    with open(RUTA, encoding="utf-8") as f:
        return yaml.safe_load(f)


def uri() -> str:
    """URI del servidor de seguimiento.

    La variable de entorno MLFLOW_TRACKING_URI, si existe, tiene prioridad sobre
    el archivo. Sirve para apuntar a otro servidor sin editar config.yaml.
    """
    del_entorno = os.environ.get("MLFLOW_TRACKING_URI")
    if del_entorno:
        return del_entorno

    cfg = cargar()["mlflow"]
    return f"http://{cfg['host']}:{cfg['puerto']}"


def conectar(verificar: bool = True) -> str:
    """Apunta MLflow al servidor del EC2. Devuelve la URI.

    Con verificar=True falla de inmediato si el servidor no responde. Sin esa
    comprobacion MLflow no da error: escribe las corridas en un ./mlruns local
    y la UI del EC2 aparece vacia sin explicacion.
    """
    destino = uri()

    if verificar and destino.startswith("http"):
        try:
            urllib.request.urlopen(f"{destino}/health", timeout=10)
        except Exception as e:
            raise ConnectionError(
                f"No hay respuesta de MLflow en {destino}\n"
                f"  ({type(e).__name__}: {e})\n\n"
                "Causa mas probable: la IP del EC2 cambio. Cambia cada vez que la\n"
                "instancia se detiene y se vuelve a encender.\n\n"
                "Avisa al grupo para que confirmen la IP actual y actualicen el\n"
                "campo 'host' de config.yaml. Si tu IP de red cambio (otro wifi,\n"
                "VPN), tambien puede ser que el puerto 8050 ya no este abierto\n"
                "para ti en el security group."
            ) from e

    mlflow.set_tracking_uri(destino)
    return destino


if __name__ == "__main__":
    print(f"Conectado a {conectar()}")
