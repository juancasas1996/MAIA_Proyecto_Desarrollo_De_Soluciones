"""
Exploracion de datos - Sleep-EDF Database Expanded
Proyecto: Clasificacion automatica de estadios del sueno
Autor: Isabella Camargo
"""

import mne
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUTA_PSG = "data/sleep-cassette/SC4001E0-PSG.edf"
RUTA_HIPNOGRAMA = "data/sleep-cassette/SC4001EC-Hypnogram.edf"

print("Cargando encabezado de la senal PSG (sin cargar toda la data en memoria)...")
raw = mne.io.read_raw_edf(RUTA_PSG, preload=False, verbose=False)

print("\n--- INFO GENERAL DE LA SENAL ---")
print(f"Canales disponibles: {raw.ch_names}")
print(f"Frecuencia de muestreo: {raw.info['sfreq']} Hz")
print(f"Duracion total: {raw.times[-1] / 3600:.2f} horas")
print(f"Numero de muestras: {raw.n_times}")

print("\nCargando hipnograma...")
anotaciones = mne.read_annotations(RUTA_HIPNOGRAMA)
raw.set_annotations(anotaciones, emit_warning=False)

print("\n--- ETAPAS DE SUENO ENCONTRADAS ---")
descripciones = anotaciones.description
etapas_unicas = sorted(set(descripciones))
for etapa in etapas_unicas:
    cantidad = list(descripciones).count(etapa)
    print(f"{etapa}: {cantidad} anotaciones")

print("\nCargando solo un fragmento de 30 segundos para graficar...")
raw_fragmento = raw.copy().crop(tmin=0, tmax=30)
raw_fragmento.load_data()

fig = raw_fragmento.plot(duration=30, n_channels=len(raw.ch_names), show=False)
fig.savefig("fragmento_senal.png", dpi=100)
print("Grafica guardada como fragmento_senal.png")

conteos = [list(descripciones).count(e) for e in etapas_unicas]

plt.figure(figsize=(8, 5))
plt.bar(etapas_unicas, conteos, color="steelblue")
plt.xlabel("Etapa de sueno")
plt.ylabel("Numero de epocas anotadas")
plt.title("Distribucion de etapas de sueno - Sujeto SC4001")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("distribucion_etapas.png", dpi=100)
print("Grafica guardada como distribucion_etapas.png")

print("\nExploracion completada.")
