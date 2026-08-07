"""
Etapa 1 del pipeline: SUBMUESTREO Y PARTICIONES
------------------------------------------------------------
Una sola responsabilidad: decidir que imagenes se usan y a que particion pertenece cada una.

  Lee : data/raw/asl_alphabet_train/  (29 carpetas de 3000 jpg)
  Hace: - valida que el dataset crudo este completo
        - toma una submuestra aleatoria de IMGS_POR_CLASE por clase
        - reparte cada clase en train/val/test de forma estratificada
  Escribe: data/processed/01_particiones.csv

No copia ni modifica imagenes: solo produce el inventario de rutas.
Las rutas se guardan relativas a la raiz para que el csv sirva en cualquier maquina.

Ejecutar:  python src/01_particiones.py
"""

import numpy as np
import pandas as pd

import config as cfg
from utils import afirmar, banner, guardar_df

EXTENSION = "*.jpg"


def listar_crudo():
    """Devuelve {clase: [rutas ordenadas]} leyendo las carpetas del dataset crudo."""
    por_clase = {}
    for clase in cfg.CLASES:
        dir_clase = cfg.RUTA_TRAIN_CRUDO / clase
        por_clase[clase] = sorted(dir_clase.glob(EXTENSION))
    return por_clase


def validar_crudo(por_clase):
    """Contrato de entrada: 29 clases, todas con 3000 imagenes."""
    faltantes = [c for c, rutas in por_clase.items() if not rutas]
    afirmar(not faltantes, f"las {cfg.NUM_CLASES} clases existen y tienen imagenes")

    incompletas = {c: len(r) for c, r in por_clase.items() if len(r) != cfg.IMGS_POR_CLASE_CRUDO}
    if incompletas:
        print(f"[FALLO] clases con conteo distinto de {cfg.IMGS_POR_CLASE_CRUDO}: {incompletas}")
        print("        la copia del dataset se trunco, vuelve a extraer el zip completo")
    afirmar(not incompletas, f"todas las clases tienen {cfg.IMGS_POR_CLASE_CRUDO} imagenes")

    total = sum(len(r) for r in por_clase.values())
    afirmar(total == cfg.TOTAL_CRUDO, f"total de imagenes crudas = {total}")


def submuestrear_y_partir(por_clase):
    """Toma IMGS_POR_CLASE por clase y las reparte en train/val/test."""
    rng = np.random.default_rng(cfg.SEMILLA)

    n_train = int(cfg.IMGS_POR_CLASE * cfg.PROP_TRAIN)
    n_val = int(cfg.IMGS_POR_CLASE * cfg.PROP_VAL)
    print(f"[info] por clase: {n_train} train, {n_val} val, {cfg.IMGS_POR_CLASE - n_train - n_val} test")

    filas = []
    for clase in cfg.CLASES:
        rutas = por_clase[clase]
        elegidas = rng.choice(len(rutas), size=cfg.IMGS_POR_CLASE, replace=False)
        rng.shuffle(elegidas)

        for orden, idx in enumerate(elegidas):
            if orden < n_train:
                particion = "train"
            elif orden < n_train + n_val:
                particion = "val"
            else:
                particion = "test"

            filas.append({
                "ruta": rutas[idx].relative_to(cfg.RAIZ).as_posix(),
                "clase": clase,
                "particion": particion,
            })

    return pd.DataFrame(filas)


def validar_salida(df):
    """Contrato de salida: sin duplicados, estratificado y del tamaño esperado."""
    esperado = cfg.NUM_CLASES * cfg.IMGS_POR_CLASE
    afirmar(len(df) == esperado, f"la submuestra tiene {len(df)} filas")
    afirmar(df["ruta"].nunique() == len(df), "no hay rutas duplicadas")
    afirmar(df["clase"].nunique() == cfg.NUM_CLASES, f"estan las {cfg.NUM_CLASES} clases")

    conteo = df.groupby(["clase", "particion"]).size().unstack(fill_value=0)
    afirmar(conteo.nunique().eq(1).all(), "todas las clases tienen el mismo reparto train/val/test")

    print("")
    print(df["particion"].value_counts().rename("imagenes").to_frame().to_string())


def main():
    banner("etapa 1: submuestreo y particiones")

    por_clase = listar_crudo()
    validar_crudo(por_clase)

    df = submuestrear_y_partir(por_clase)
    validar_salida(df)

    guardar_df(df, cfg.RUTA_PARTICIONES)


main()
