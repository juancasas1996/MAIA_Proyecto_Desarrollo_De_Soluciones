#!/usr/bin/env bash
# Levanta el servidor de MLflow en el EC2, dentro de una sesion de tmux.
#
# Resuelve solo la IP publica desde los metadatos de la instancia, que es lo
# que cambia en cada Stop/Start y obliga a reescribir el comando a mano.
#
# Uso:  ./start-mlflow.sh

set -uo pipefail

VENV="/home/ubuntu/.venv"
DIR="/home/ubuntu"          # el backend sqlite es una ruta RELATIVA: siempre desde aqui
PUERTO=8050
SESION="mlflow"
LOG="${DIR}/mlflow.log"

# --- Ya esta corriendo? ------------------------------------------------------
if ss -ltn 2>/dev/null | grep -q ":${PUERTO} "; then
    echo "MLflow ya esta escuchando en el puerto ${PUERTO}."
    echo "  Ver salida:  tmux attach -t ${SESION}"
    exit 0
fi

# Sesion de tmux huerfana: existe pero el servidor de adentro murio
if tmux has-session -t "${SESION}" 2>/dev/null; then
    echo "Limpiando sesion tmux previa..."
    tmux kill-session -t "${SESION}"
fi

# --- IP publica actual -------------------------------------------------------
TOKEN=$(curl -sf -X PUT "http://169.254.169.254/latest/api/token" \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 300" 2>/dev/null)
IP=$(curl -sf -H "X-aws-ec2-metadata-token: ${TOKEN}" \
    "http://169.254.169.254/latest/meta-data/public-ipv4" 2>/dev/null)

if [ -z "${IP}" ]; then
    echo "ERROR: no se pudo leer la IP publica desde los metadatos de la instancia."
    echo "Verifica que estas ejecutando esto DENTRO del EC2."
    exit 1
fi

# --- Lanzar ------------------------------------------------------------------
tmux new-session -d -s "${SESION}" -c "${DIR}" \
    "${VENV}/bin/mlflow server -h 0.0.0.0 -p ${PUERTO} \
        --allowed-hosts localhost:${PUERTO},${IP}:${PUERTO} \
        --cors-allowed-origins http://${IP}:${PUERTO} 2>&1 | tee -a ${LOG}"

# --- Esperar a que responda --------------------------------------------------
printf "Arrancando"
for _ in $(seq 1 20); do
    if ss -ltn 2>/dev/null | grep -q ":${PUERTO} "; then
        printf "\n\n  MLflow arriba.\n\n"
        printf "  UI en el navegador:\n     http://%s:%s\n\n" "${IP}" "${PUERTO}"
        printf "  Desde tu Mac, antes de correr experimentos:\n"
        printf "     export MLFLOW_TRACKING_URI=\"http://%s:%s\"\n\n" "${IP}" "${PUERTO}"
        printf "  Ver la salida:  tmux attach -t %s    (Ctrl+B, luego D para salir)\n" "${SESION}"
        printf "  Detenerlo:      ./stop-mlflow.sh\n"
        exit 0
    fi
    printf "."
    sleep 1
done

printf "\nERROR: no arranco en 20 segundos.\n"
echo "Revisa el log:  tail -50 ${LOG}"
exit 1
