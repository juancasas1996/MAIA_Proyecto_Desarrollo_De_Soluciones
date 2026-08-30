# SomnoAI — Estimación de la edad cerebral a partir del EEG del sueño

Proyecto - Desarrollo de Soluciones
Maestría en Inteligencia Artificial, Universidad de los Andes

## El problema

La actividad eléctrica cerebral durante el sueño cambia de forma sistemática con
el envejecimiento. SomnoAI estima una **edad cerebral** a partir del EEG de una
polisomnografía nocturna y calcula el **Brain Age Index** (BAI), la diferencia
entre esa estimación y la edad cronológica del sujeto.

El BAI sirve como criterio automático de priorización: permite a un especialista
en sueño revisar primero los registros con mayor divergencia. Es una herramienta
de apoyo analítico, no un instrumento de diagnóstico clínico.

## Los datos

[Sleep-EDF Database Expanded](https://physionet.org/content/sleep-edfx/1.0.0/),
subconjunto **Sleep Cassette**: 153 registros nocturnos de 78 sujetos sanos sin
medicación, entre 25 y 101 años.

Los datos se versionan con **DVC** sobre S3 y no viven en Git. Para obtenerlos:

```bash
dvc pull
```

## Estructura

```
Data/            Datos crudos (DVC → S3). No versionados en Git.
EDA/             Análisis exploratorio — Entrega 1
                 EDA_Consolidado.ipynb reúne el análisis del equipo
Maqueta/         Maqueta del tablero, iteraciones v1 a v5
Experimentos/    Modelado y seguimiento con MLflow — Entrega 2
  config.py      conectar() — apunta MLflow al servidor y verifica que responda
  config.yaml    host y puerto del EC2. La IP cambia en cada arranque
  subject_split_seed42.json   la partición del proyecto: NO se regenera
  <Nombre>/      Una carpeta por integrante
```

## Puesta en marcha

```bash
pip install -r requirements.txt
dvc pull
```

Los experimentos se registran en un servidor de MLflow alojado en EC2. La
configuración de conexión está en `Experimentos/config.yaml`, y `conectar()`
de `Experimentos/config.py` apunta MLflow a ese servidor:

```python
import sys; sys.path.append("..")
from config import conectar
conectar()
```

## Hallazgos que orientan el modelado

Del análisis exploratorio ([`EDA/EDA_Consolidado.ipynb`](EDA/EDA_Consolidado.ipynb)):

- Solo los dos canales de EEG y el de EOG se registraron a 100 Hz. El resto está
  a 1 Hz y no permite observar los husos de sueño (12–16 Hz).
- El periodo de sueño ocupa cerca de un tercio del registro: los registros son
  ambulatorios y cubren un ciclo día-noche completo.
- El porcentaje de N1 es la variable con mayor correlación individual con la edad
  (r = +0,64), por encima de cualquier característica espectral.
- La potencia en la banda sigma disminuye con la edad (r = −0,50).
- **Línea base**: una regresión Ridge sobre arquitectura del sueño y
  características espectrales estima la edad con un MAE de 10,2 años, frente a
  los 18,3 de predecir siempre la edad media.

## Estado del modelado

Un Ridge con corrección del sesgo del BAI sobre los dos canales de EEG estima la
edad con un **MAE de 10,43 años en validación cruzada** (9,15 sobre los 16
sujetos de test, cifra optimista porque el test se ha consultado varias veces).
Predecir siempre la edad media cuesta 19,10.

Detalle, decisiones y limitaciones en
[`Experimentos/Sebastian/`](Experimentos/Sebastian/).

## Equipo

- Isabella del Pilar Camargo Salazar
- Javier Andrés Marín Gallón
- Diego Charry Cárdenas
- Juan Camilo Martínez Vélez
- Juan Sebastián Casas Castillo
