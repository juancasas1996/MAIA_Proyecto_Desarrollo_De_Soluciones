import mne
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import welch
from mne.datasets.sleep_physionet.age import fetch_data


def encontrar_raiz_repositorio():
    inicio_script = Path(__file__).resolve().parent
    candidatos = [Path.cwd(), *Path.cwd().parents, inicio_script, *inicio_script.parents]
    for ruta in candidatos:
        if (ruta / '.git').exists():
            return ruta.resolve()
    raise FileNotFoundError('No se encontro la raiz del repositorio Git.')


# ============================================
# PASO 1: Descargar 8 sujetos (indices 1 al 8)
# ============================================
raiz = encontrar_raiz_repositorio()
indices_sujetos = [1, 2, 3, 4, 5, 6, 7, 8]
print('Descargando sujetos...')
archivos_descargados = fetch_data(subjects=indices_sujetos, recording=[1])
print('Descarga completa.')

# ============================================
# PASO 2: Leer la tabla de edades
# ============================================
tabla_edades = pd.read_excel(raiz / 'Data' / 'SC-subjects.xls')

def obtener_edad(indice_sujeto):
    fila = tabla_edades[tabla_edades['subject'] == indice_sujeto]
    return int(fila['age'].iloc[0])

# ============================================
# PASO 3: Procesar cada sujeto (features por epoca)
# ============================================
BANDAS = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30)}

filas = []
for i, indice in enumerate(indices_sujetos):
    ruta_psg = archivos_descargados[i][0]  # el archivo -PSG.edf
    edad = obtener_edad(indice)
    codigo_sujeto = f'SC{indice:03d}'

    print(f'Procesando {codigo_sujeto} (edad {edad})...')
    raw = mne.io.read_raw_edf(ruta_psg, preload=True, verbose=False)
    epochs = mne.make_fixed_length_epochs(raw, duration=30.0, preload=True, verbose=False)
    data = epochs.get_data()
    sfreq = raw.info['sfreq']

    for epoca in data:
        fila = {}
        for canal_idx in range(epoca.shape[0]):
            senal = epoca[canal_idx]
            freqs, psd = welch(senal, fs=sfreq, nperseg=min(len(senal), int(sfreq*2)))
            for nombre_banda, (fmin, fmax) in BANDAS.items():
                mask = (freqs >= fmin) & (freqs <= fmax)
                potencia = np.trapz(psd[mask], freqs[mask])
                fila[f'canal{canal_idx}_{nombre_banda}'] = potencia
        fila['sujeto'] = codigo_sujeto
        fila['edad'] = edad
        filas.append(fila)

df = pd.DataFrame(filas)
archivo_salida = raiz / 'Experimentos' / 'Isabella' / 'datos' / 'features_eeg.csv'
archivo_salida.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(archivo_salida, index=False)
print(f'Listo! Tabla guardada como {archivo_salida}')
print(df.shape)
print(df['sujeto'].value_counts())
