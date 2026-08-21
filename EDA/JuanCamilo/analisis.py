"""
Análisis complementario de balance de clases - Sleep-EDF Expanded
Proyecto: Clasificación automática de estadios del sueño
Autor: Juan Camilo Martínez Vélez

Aporte:
- Calcula la duración real de cada etapa dentro de la ventana de sueño.
- Convierte las duraciones a épocas equivalentes de 30 segundos.
- Recorta vigilia prolongada antes/después del sueño y compara porcentajes.
- Genera un CSV reproducible y dos visualizaciones.

Este enfoque complementa el conteo de anotaciones, porque una anotación puede
representar más de una época de 30 segundos.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne


MAX_SUJETOS = 10
DURACION_EPOCA_SEGUNDOS = 30.0
MARGEN_VIGILIA_SEGUNDOS = 30 * 60
DIRECTORIO_SCRIPT = Path(__file__).resolve().parent
RAIZ_REPOSITORIO = DIRECTORIO_SCRIPT.parents[1]
DIRECTORIO_RESULTADOS = DIRECTORIO_SCRIPT / "resultados"


def localizar_directorio_datos() -> Path:
    """Localiza la carpeta del dataset desde la raíz del repositorio."""
    candidatos = [
        RAIZ_REPOSITORIO / "Data" / "sleep-cassette",
        RAIZ_REPOSITORIO / "data" / "sleep-cassette",
    ]

    for candidato in candidatos:
        if candidato.exists():
            return candidato

    rutas = ", ".join(str(ruta) for ruta in candidatos)
    raise FileNotFoundError(
        "No se encontró el dataset Sleep-EDF. "
        f"Se buscaron estas rutas: {rutas}."
    )


def obtener_codigo_sujeto(nombre_archivo: str) -> str:
    """Extrae el identificador base, por ejemplo SC4001."""
    return nombre_archivo.split("E", maxsplit=1)[0]


def descubrir_pares(directorio: Path, limite: int) -> list[tuple[str, Path, Path]]:
    """Relaciona cada hipnograma con su archivo PSG correspondiente."""
    pares: list[tuple[str, Path, Path]] = []

    for hipnograma in sorted(directorio.glob("*-Hypnogram.edf")):
        codigo = obtener_codigo_sujeto(hipnograma.name)
        candidatos_psg = sorted(directorio.glob(f"{codigo}*-PSG.edf"))

        if not candidatos_psg:
            print(f"Advertencia: no se encontró PSG para {hipnograma.name}")
            continue

        pares.append((codigo, candidatos_psg[0], hipnograma))

        if len(pares) >= limite:
            break

    if not pares:
        raise FileNotFoundError(
            f"No se encontraron pares PSG/Hypnograma en {directorio}."
        )

    return pares


def es_etapa_valida(etapa: str) -> bool:
    """Excluye etiquetas desconocidas y eventos que no son etapas de sueño."""
    return etapa.startswith("Sleep stage") and etapa != "Sleep stage ?"


def analizar_hipnograma(hipnograma: Path) -> dict[str, float]:
    """
    Suma la duración efectiva de cada etapa dentro de la ventana de sueño.

    Sleep-EDF contiene largos periodos de vigilia antes y después del sueño.
    Para evitar que esos periodos dominen la distribución, se conserva desde
    30 minutos antes de la primera etapa no-W hasta 30 minutos después de la
    última etapa no-W.
    """
    anotaciones = mne.read_annotations(hipnograma)

    eventos = [
        (str(etapa), float(inicio), float(duracion))
        for etapa, inicio, duracion in zip(
            anotaciones.description,
            anotaciones.onset,
            anotaciones.duration,
        )
    ]

    eventos_sueno = [
        (etapa, inicio, duracion)
        for etapa, inicio, duracion in eventos
        if etapa.startswith("Sleep stage")
        and etapa not in {"Sleep stage W", "Sleep stage ?"}
    ]

    if not eventos_sueno:
        raise ValueError(
            f"No se encontraron etapas de sueño válidas en {hipnograma.name}."
        )

    inicio_sueno = min(inicio for _, inicio, _ in eventos_sueno)
    fin_sueno = max(inicio + duracion for _, inicio, duracion in eventos_sueno)

    inicio_ventana = max(0.0, inicio_sueno - MARGEN_VIGILIA_SEGUNDOS)
    fin_ventana = fin_sueno + MARGEN_VIGILIA_SEGUNDOS

    duraciones: dict[str, float] = defaultdict(float)

    for etapa, inicio, duracion in eventos:
        if not etapa.startswith("Sleep stage"):
            continue

        fin = inicio + duracion
        inicio_solapado = max(inicio, inicio_ventana)
        fin_solapado = min(fin, fin_ventana)
        duracion_solapada = max(0.0, fin_solapado - inicio_solapado)

        if duracion_solapada > 0:
            duraciones[etapa] += duracion_solapada

    return dict(duraciones)


def guardar_csv(
    resultados: dict[str, dict[str, float]],
    etapas_validas: list[str],
    archivo_salida: Path,
) -> None:
    """Guarda duración, épocas equivalentes y porcentaje por sujeto."""
    with archivo_salida.open("w", newline="", encoding="utf-8") as archivo:
        campos = [
            "sujeto",
            "etapa",
            "duracion_segundos",
            "duracion_minutos",
            "epocas_equivalentes_30s",
            "porcentaje_etapas_validas",
        ]
        escritor = csv.DictWriter(archivo, fieldnames=campos)
        escritor.writeheader()

        for sujeto, duraciones in resultados.items():
            total_valido = sum(duraciones.get(etapa, 0.0) for etapa in etapas_validas)

            for etapa in etapas_validas:
                segundos = duraciones.get(etapa, 0.0)
                porcentaje = 100.0 * segundos / total_valido if total_valido else 0.0

                escritor.writerow(
                    {
                        "sujeto": sujeto,
                        "etapa": etapa,
                        "duracion_segundos": round(segundos, 2),
                        "duracion_minutos": round(segundos / 60.0, 2),
                        "epocas_equivalentes_30s": round(
                            segundos / DURACION_EPOCA_SEGUNDOS, 2
                        ),
                        "porcentaje_etapas_validas": round(porcentaje, 4),
                    }
                )


def graficar_porcentajes_por_sujeto(
    resultados: dict[str, dict[str, float]],
    etapas_validas: list[str],
    archivo_salida: Path,
) -> None:
    """Genera barras apiladas con la distribución porcentual por sujeto."""
    sujetos = list(resultados)
    acumulado = [0.0] * len(sujetos)

    fig, ax = plt.subplots(figsize=(12, 7))

    for etapa in etapas_validas:
        porcentajes = []

        for sujeto in sujetos:
            duraciones = resultados[sujeto]
            total_valido = sum(
                duraciones.get(etapa_actual, 0.0)
                for etapa_actual in etapas_validas
            )
            valor = duraciones.get(etapa, 0.0)
            porcentajes.append(100.0 * valor / total_valido if total_valido else 0.0)

        ax.bar(sujetos, porcentajes, bottom=acumulado, label=etapa)
        acumulado = [
            base + porcentaje
            for base, porcentaje in zip(acumulado, porcentajes)
        ]

    ax.set_xlabel("Sujeto")
    ax.set_ylabel("Porcentaje del tiempo anotado (%)")
    ax.set_title("Distribución porcentual de etapas de sueño por sujeto")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Etapa", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(archivo_salida, dpi=150)
    plt.close(fig)


def graficar_distribucion_agregada(
    resultados: dict[str, dict[str, float]],
    etapas_validas: list[str],
    archivo_salida: Path,
) -> dict[str, float]:
    """Agrega todos los sujetos y grafica la distribución global."""
    agregado = {
        etapa: sum(
            duraciones.get(etapa, 0.0)
            for duraciones in resultados.values()
        )
        for etapa in etapas_validas
    }

    total = sum(agregado.values())
    porcentajes = {
        etapa: 100.0 * segundos / total if total else 0.0
        for etapa, segundos in agregado.items()
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(list(porcentajes), list(porcentajes.values()))
    ax.set_xlabel("Etapa de sueño")
    ax.set_ylabel("Porcentaje agregado (%)")
    ax.set_title("Distribución agregada de etapas de sueño")
    ax.tick_params(axis="x", rotation=45)

    for indice, porcentaje in enumerate(porcentajes.values()):
        ax.text(indice, porcentaje, f"{porcentaje:.1f}%", ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(archivo_salida, dpi=150)
    plt.close(fig)

    return porcentajes


def main() -> None:
    DIRECTORIO_RESULTADOS.mkdir(exist_ok=True)

    directorio_datos = localizar_directorio_datos()
    pares = descubrir_pares(directorio_datos, MAX_SUJETOS)

    print(f"Directorio de datos: {directorio_datos}")
    print(f"Sujetos seleccionados: {len(pares)}")

    resultados: dict[str, dict[str, float]] = {}

    for codigo, archivo_psg, archivo_hipnograma in pares:
        print(f"\nProcesando {codigo}...")

        raw = mne.io.read_raw_edf(archivo_psg, preload=False, verbose=False)
        duracion_psg_horas = raw.times[-1] / 3600.0
        duraciones = analizar_hipnograma(archivo_hipnograma)

        resultados[codigo] = duraciones

        print(f"  PSG: {archivo_psg.name}")
        print(f"  Duración PSG: {duracion_psg_horas:.2f} horas")
        print(f"  Etiquetas encontradas: {sorted(duraciones)}")

    etapas_validas = sorted(
        {
            etapa
            for duraciones in resultados.values()
            for etapa in duraciones
            if es_etapa_valida(etapa)
        }
    )

    archivo_csv = DIRECTORIO_RESULTADOS / "resumen_etapas_30s.csv"
    grafica_sujetos = (
        DIRECTORIO_RESULTADOS / "distribucion_porcentual_por_sujeto.png"
    )
    grafica_agregada = (
        DIRECTORIO_RESULTADOS / "distribucion_agregada_etapas.png"
    )

    guardar_csv(resultados, etapas_validas, archivo_csv)
    graficar_porcentajes_por_sujeto(
        resultados,
        etapas_validas,
        grafica_sujetos,
    )
    porcentajes_agregados = graficar_distribucion_agregada(
        resultados,
        etapas_validas,
        grafica_agregada,
    )

    valores_positivos = [
        valor for valor in porcentajes_agregados.values() if valor > 0
    ]
    razon_desbalance = (
        max(valores_positivos) / min(valores_positivos)
        if valores_positivos
        else 0.0
    )

    print("\n=== RESUMEN AGREGADO ===")
    for etapa, porcentaje in porcentajes_agregados.items():
        print(f"{etapa}: {porcentaje:.2f}%")

    print(f"Razón de desbalance máximo/mínimo: {razon_desbalance:.2f}")
    print(f"CSV generado: {archivo_csv}")
    print(f"Gráfica por sujeto: {grafica_sujetos}")
    print(f"Gráfica agregada: {grafica_agregada}")
    print("\nAnálisis completado.")


if __name__ == "__main__":
    main()
