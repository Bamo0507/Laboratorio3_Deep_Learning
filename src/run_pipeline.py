"""
Orquestador del pipeline
------------------------------------------------------------
Una sola responsabilidad: correr las etapas en orden y detenerse en la primera que falle.

  Hace: - ejecuta cada script de ETAPAS con el python del venv
        - si una etapa devuelve codigo distinto de 0, corta ahi y reporta cual fallo

No re-implementa logica: solo encadena scripts que ya funcionan por separado.
La etapa 0 (00_init.py) queda fuera porque se corre con el python del sistema,
antes de que exista el venv.

Ejecutar:  python src/run_pipeline.py
"""

import subprocess
import sys
from pathlib import Path

DIR_SRC = Path(__file__).resolve().parent

ETAPAS = [
    "01_particiones.py",
    "02_preprocesamiento.py",
    "03_augmentation.py",
]


def anunciar(mensaje):
    """Imprime vaciando el buffer, para no desordenar el log frente a los subprocesos."""
    print(mensaje, flush=True)


def main():
    anunciar("=" * 60)
    anunciar(f"PIPELINE COMPLETO: {len(ETAPAS)} ETAPAS")
    anunciar("=" * 60)

    for i, etapa in enumerate(ETAPAS, 1):
        anunciar(f"\n>>> [{i}/{len(ETAPAS)}] {etapa}\n")
        resultado = subprocess.run([sys.executable, str(DIR_SRC / etapa)])

        if resultado.returncode != 0:
            anunciar("")
            anunciar("=" * 60)
            anunciar(f"[FALLO] el pipeline se detuvo en {etapa} (codigo {resultado.returncode})")
            anunciar("=" * 60)
            sys.exit(1)

    anunciar("")
    anunciar("=" * 60)
    anunciar("[ok] pipeline completo sin errores")
    anunciar("=" * 60)


main()
