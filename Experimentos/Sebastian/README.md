# Modelado — Juan Sebastián Casas

Tres notebooks, en orden. Cada uno se ejecuta de arriba abajo.

| | Notebook | Qué hace | Tarda |
|---|---|---|---|
| 01 | `01_particion` | fija qué 62 sujetos son desarrollo y qué 16 son test | segundos |
| 02 | `02_caracteristicas` | 153 archivos EDF → tabla de características, los dos canales | 2,5 min |
| 03 | `03_modelo` | Ridge con corrección del BAI · Fpz-Cz vs Pz-Oz vs los dos | 15 s + 5 min |

El notebook 03 necesita el servidor de MLflow arriba: los 5 minutos son el registro de las
276 corridas, no el ajuste. Si el EC2 no responde, `conectar()` falla en la primera celda.

La partición va **primera y a propósito**: se decide leyendo solo `SC-subjects.xls`, antes
de calcular nada de la señal, así no puede estar influida por las características.

## El resultado

MAE en años sobre los 16 sujetos de test. Predecir siempre la edad media da **19,10**.

| Experimento | Columnas | train | validación | test | r(BAI, edad) |
|---|---:|---:|---:|---:|---:|
| A · solo Fpz-Cz | 20 | 8,37 | 10,36 | 10,85 | −0,46 |
| B · solo Pz-Oz | 20 | 8,14 | 11,62 | 14,59 | −0,29 |
| **C · los dos** | **32** | **7,48** | **10,43** | **9,15** | **−0,48** |

Dos avisos para el reporte:

- **La cifra defendible es 10,43**, la de validación cruzada. El 9,15 son 16 sujetos y el
  test se ha consultado muchas veces a lo largo del proyecto, así que está sesgado a la baja.
- **La corrección del sesgo del BAI no generaliza.** Los tres experimentos cumplen
  `|r| < 0,25` en validación cruzada y ninguno en test. Es tamaño de muestra: Sun et al.
  aplican el mismo método con 1.343 sujetos de entrenamiento; aquí hay 62.

## Datos

```
datos/caracteristicas_noche_FpzCz.csv    153 noches × 97 columnas   (lo produce 02)
datos/caracteristicas_noche_PzOz.csv     153 noches × 97 columnas   (lo produce 02)
../subject_split_seed42.json             la partición               (lo produce 01)
```

`01_particion` **no sobrescribe** la partición si ya existe: la regenera, la compara y
aborta si difiere. Todos los experimentos del proyecto reutilizan ese archivo; si cada uno
sortea el suyo, las comparaciones dejan de valer y el test acaba contaminado.

## Qué queda en MLflow

Un solo experimento, `edad-cerebral-sebastian`, con **276 corridas**: una padre por canal
y las 91 combinaciones de su rejilla (13 `alpha` × 7 `lambda`) anidadas debajo.

```
edad-cerebral-sebastian
├── A · Fpz-Cz    →  91 anidadas
├── B · Pz-Oz     →  91 anidadas
└── C · los dos   →  91 anidadas   → Registry: edad-cerebral v1
```

Un experimento y no tres porque MLflow compara **dentro** de un experimento: separarlos
impediría poner A, B y C en la misma gráfica, que es el objetivo del notebook.

Cada padre registra los dos CSV como *datasets*, los hiperparámetros y el criterio de
selección, las métricas de train/validación/test con su referencia de predecir la media,
las tres figuras, la rejilla completa y la partición, y el modelo serializado.

Para verlas en plano y compararlas, en la caja de búsqueda de la UI:

```
tags.canal = 'C'        las 91 de un canal
tags.tipo = 'rejilla'   las 273 de los tres
```

Sin ese filtro la tabla solo muestra las tres padre, con las anidadas plegadas bajo el
`⊞` de cada una.

## Por qué el modelo es una regresión lineal

Antes de fijar el Ridge del notebook 03 se probaron **ocho familias de modelos** —kernel
de grado 2, PLS, ElasticNet, el modelo fundacional LUNA, redes neuronales, stacking por
etapa del sueño y búsqueda automática de modelo—. **Ninguna mejora la recta.** El mejor
alternativo (PLS, 9,30) empata dentro del ruido; el resto pierde entre 1 y 5 años.

El modo de fallo no fue el sesgo sino la varianza: con 62 sujetos, cada grado de libertad
extra —más columnas, más componentes, más modelos que elegir— separa la validación del
test. El stacking es el caso extremo: 8,22 en desarrollo y 14,21 en test.

Esos nueve notebooks, sus resultados y el análisis completo están **fuera del repositorio**,
en `Microproyectos/Exploracion_Modelos/` (ver su `LEEME.txt`). No hacen falta para
reproducir nada de lo de arriba.

## Reglas del proyecto

- **La unidad es el sujeto**, nunca la noche ni la época. Las dos noches de una persona se
  promedian antes de modelar.
- **La partición no se regenera.**
- **El test se mira una vez por experimento**, al final. Cualquier cifra nueva sobre él va
  etiquetada como *post-hoc*.
- **Ningún criterio de selección se aplica sin filtrar antes por «que bata a predecir la
  media».** Sin ese filtro, elegir por sesgo del BAI escoge modelos que interpolan el
  entrenamiento.
