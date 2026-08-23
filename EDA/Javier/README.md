# EDA - Javier Andres Marin Gallon

`Analisis_EOG_EMG_Transiciones_Javier.ipynb`

Explora tres cosas que no cubrian los EDA anteriores: que aportan el EOG y el EMG, la estructura
temporal del hipnograma, y como partir los datos sin fuga de informacion.

## Hallazgos

- El EMG esta muestreado a **1 Hz**, no a 100 Hz. No admite analisis espectral.
- El EMG separa **REM de N1** con AUC ~0.90; ninguna otra caracteristica pasa de ~0.61.
  Configuracion recomendada: **EEG+EMG**.
- El **89.8%** de las epocas repiten el estadio anterior. N1 y N3 son fragmentados; REM es sostenido.
- Partir por epoca deja al **97.5%** de la prueba con una vecina en entrenamiento. Se incluye una
  particion por sujeto, estratificada y verificada.
- `?` es cola no anotada y se descarta. `M` son solo 128 epocas: mejor tratarla como rechazo en
  inferencia que como sexta clase.

## Ejecucion

```bash
pip install -r ../../requirements.txt
dvc pull
jupyter lab Analisis_EOG_EMG_Transiciones_Javier.ipynb
```

Las secciones 2, 4 y 5 usan todos los registros; la seccion 3 se limita a 12 por costo de lectura.
