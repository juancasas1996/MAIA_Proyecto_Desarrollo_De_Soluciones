"""
Exploracion de datos - Sleep-EDF Database Expanded
Proyecto: Clasificacion automatica de estadios del sueno
Autor: Isabella Camargo

Compara 2 sujetos para revisar si el desbalance de clases es un
patron general del dataset o algo particular de un solo sujeto.
"""

import mne
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Este script vive en EDA/Isabella/, por lo que los datos están dos niveles arriba
DATA_DIR = "../../Data/sleep-cassette"

SUJETOS = [
    {
        "nombre": "SC4001",
        "psg": f"{DATA_DIR}/SC4001E0-PSG.edf",
        "hipnograma": f"{DATA_DIR}/SC4001EC-Hypnogram.edf",
    },
    {
        "nombre": "SC4002",
        "psg": f"{DATA_DIR}/SC4002E0-PSG.edf",
        "hipnograma": f"{DATA_DIR}/SC4002EC-Hypnogram.edf",
    },
]

resultados = {}

for sujeto in SUJETOS:
    nombre = sujeto["nombre"]
    print(f"\n===== Procesando sujeto {nombre} =====")

    raw = mne.io.read_raw_edf(sujeto["psg"], preload=False, verbose=False)
    print(f"Canales: {raw.ch_names}")
    print(f"Duracion total: {raw.times[-1] / 3600:.2f} horas")

    anotaciones = mne.read_annotations(sujeto["hipnograma"])
    descripciones = anotaciones.description
    etapas_unicas = sorted(set(descripciones))

    conteo_etapas = {}
    for etapa in etapas_unicas:
        conteo_etapas[etapa] = list(descripciones).count(etapa)
        print(f"  {etapa}: {conteo_etapas[etapa]} anotaciones")

    resultados[nombre] = conteo_etapas

# -----------------------------------------------------------------
# Graficar comparacion de ambos sujetos en una sola figura
# -----------------------------------------------------------------
print("\nGenerando grafica comparativa...")

todas_etapas = sorted(set().union(*[r.keys() for r in resultados.values()]))

fig, ax = plt.subplots(figsize=(10, 6))
ancho_barra = 0.35
posiciones = range(len(todas_etapas))

for i, (nombre, conteo) in enumerate(resultados.items()):
    valores = [conteo.get(etapa, 0) for etapa in todas_etapas]
    offset = [p + i * ancho_barra for p in posiciones]
    ax.bar(offset, valores, width=ancho_barra, label=nombre)

ax.set_xlabel("Etapa de sueno")
ax.set_ylabel("Numero de epocas anotadas")
ax.set_title("Comparacion de distribucion de etapas de sueno entre sujetos")
ax.set_xticks([p + ancho_barra / 2 for p in posiciones])
ax.set_xticklabels(todas_etapas, rotation=45)
ax.legend()
plt.tight_layout()
plt.savefig("comparacion_etapas_2_sujetos.png", dpi=100)
print("Grafica guardada como comparacion_etapas_2_sujetos.png")

print("\nExploracion completada.")
