# Experimentos Isabella

Esta carpeta contiene los scripts de modelado de Isabella.

## Flujo

Desde la raiz del repositorio:

```bash
python Experimentos/Isabella/procesar_eeg.py
python Experimentos/Isabella/entrenar_modelo.py
python Experimentos/Isabella/entrenar_random_forest.py
```

`procesar_eeg.py` lee `Data/SC-subjects.xls`, descarga los registros usados por
el script y genera `Experimentos/Isabella/datos/features_eeg.csv`.

`entrenar_modelo.py` entrena el modelo Ridge con
`Experimentos/Isabella/datos/features_eeg.csv`.

`entrenar_random_forest.py` reutiliza las caracteristicas y la particion del
experimento de Sebastian:

- `Experimentos/Sebastian/datos/caracteristicas_noche_FpzCz.csv`
- `Experimentos/Sebastian/datos/caracteristicas_noche_PzOz.csv`
- `Experimentos/subject_split_seed42.json`
