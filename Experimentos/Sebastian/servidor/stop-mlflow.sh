#!/usr/bin/env bash
# Detiene el servidor de MLflow en el EC2.
#
# Mata por puerto y no por nombre de proceso: el servidor levanta varios
# workers y un patron por nombre puede dejar huerfanos escuchando.
#
# Las corridas NO se pierden: viven en mlflow.db, en disco.
#
# Uso:  ./stop-mlflow.sh

set -uo pipefail

PUERTO=8050
SESION="mlflow"

if tmux has-session -t "${SESION}" 2>/dev/null; then
    tmux kill-session -t "${SESION}"
    echo "Sesion tmux '${SESION}' cerrada."
fi

PIDS=$(sudo lsof -ti:"${PUERTO}" 2>/dev/null)
if [ -n "${PIDS}" ]; then
    echo "${PIDS}" | xargs -r sudo kill
    sleep 2
    # Si algo sobrevivio, insistir
    PIDS=$(sudo lsof -ti:"${PUERTO}" 2>/dev/null)
    [ -n "${PIDS}" ] && echo "${PIDS}" | xargs -r sudo kill -9
fi

sleep 1
if ss -ltn 2>/dev/null | grep -q ":${PUERTO} "; then
    echo "ERROR: algo sigue escuchando en el puerto ${PUERTO}."
    exit 1
fi

echo "MLflow detenido. Las corridas siguen en /home/ubuntu/mlflow.db"
