# Servidor de MLflow

Scripts que corren **dentro del EC2**, no en tu máquina. Los administra quien
gestiona la instancia; el resto del equipo no necesita entrar aquí.

## Desplegarlos al EC2

```bash
scp -i <llave>.pem Sebastian/servidor/*.sh ubuntu@<IP>:/home/ubuntu/
```

## Uso, ya dentro del EC2

```bash
./start-mlflow.sh    # levanta el servidor en tmux y muestra la IP publica
./stop-mlflow.sh     # lo baja (las corridas siguen en mlflow.db)
```

`start-mlflow.sh` resuelve la IP publica desde los metadatos de la instancia, asi
que no hay que editarla a mano tras cada Stop/Start. La IP que imprime es la que
va en el campo `host` de `../../config.yaml`.
