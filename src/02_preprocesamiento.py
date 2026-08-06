"""
Etapa 2 del pipeline: PREPROCESAMIENTO DE IMAGENES
------------------------------------------------------------
Una sola responsabilidad: convertir las rutas del inventario en tensores listos para entrenar.

  Lee : data/processed/01_particiones.csv
  Hace: - abre cada jpg y lo pasa a escala de grises
        - reescala de 200x200 a 64x64 con filtro LANCZOS
        - normaliza los pixeles de [0, 255] a [0, 1]
        - codifica la clase como indice entero segun cfg.CLASES
  Escribe: data/processed/02_tensores.npz

Ejecutar:  python src/02_preprocesamiento.py
"""

import numpy as np
from PIL import Image
from tqdm import tqdm

import config as cfg
from utils import afirmar, banner, cargar_df

MODO_GRISES = "L"
FILTRO = Image.Resampling.LANCZOS
COLUMNAS_ESPERADAS = ["ruta", "clase", "particion"]
PARTICIONES = ["train", "val", "test"]


def validar_entrada(df):
    """Contrato de entrada: lo que prometio la etapa 1."""
    afirmar(list(df.columns) == COLUMNAS_ESPERADAS, f"el csv trae las columnas {COLUMNAS_ESPERADAS}")
    afirmar(len(df) == cfg.NUM_CLASES * cfg.IMGS_POR_CLASE, f"el inventario tiene {len(df)} filas")
    afirmar(df["clase"].nunique() == cfg.NUM_CLASES, f"estan las {cfg.NUM_CLASES} clases")
    afirmar(set(df["particion"]) == set(PARTICIONES), "las particiones son train/val/test")


def procesar_imagen(ruta_relativa):
    """Abre un jpg y devuelve su matriz en grises, reescalada y normalizada."""
    with Image.open(cfg.RAIZ / ruta_relativa) as img:
        img = img.convert(MODO_GRISES).resize((cfg.RESOLUCION, cfg.RESOLUCION), FILTRO)
        return np.asarray(img, dtype=np.float32) / 255.0


def construir_tensores(df):
    """Arma el par (X, y) de cada particion recorriendo el inventario."""
    indice_clase = {clase: i for i, clase in enumerate(cfg.CLASES)}
    tensores = {}

    for particion in PARTICIONES:
        sub = df[df["particion"] == particion].reset_index(drop=True)
        x = np.zeros((len(sub), cfg.RESOLUCION, cfg.RESOLUCION, cfg.CANALES), dtype=np.float32)
        y = np.zeros(len(sub), dtype=np.int16)

        for i, fila in tqdm(sub.iterrows(), total=len(sub), desc=f"  {particion:5s}", ncols=70):
            x[i, :, :, 0] = procesar_imagen(fila["ruta"])
            y[i] = indice_clase[fila["clase"]]

        tensores[f"x_{particion}"] = x
        tensores[f"y_{particion}"] = y

    return tensores


def validar_salida(tensores):
    """Contrato de salida: formas correctas, rango normalizado y clases completas."""
    for particion in PARTICIONES:
        x = tensores[f"x_{particion}"]
        y = tensores[f"y_{particion}"]
        afirmar(x.shape[1:] == cfg.FORMA_ENTRADA, f"{particion}: cada imagen es {x.shape[1:]}")
        afirmar(len(x) == len(y), f"{particion}: {len(x)} imagenes con {len(y)} etiquetas")

    todas = np.concatenate([tensores[f"x_{p}"] for p in PARTICIONES])
    afirmar(todas.min() >= 0.0 and todas.max() <= 1.0, f"pixeles en [{todas.min():.2f}, {todas.max():.2f}]")
    afirmar(not np.isnan(todas).any(), "no hay NaN en los tensores")

    etiquetas = np.concatenate([tensores[f"y_{p}"] for p in PARTICIONES])
    afirmar(len(np.unique(etiquetas)) == cfg.NUM_CLASES, f"las {cfg.NUM_CLASES} clases estan representadas")


def main():
    banner("etapa 2: preprocesamiento de imagenes")

    df = cargar_df(cfg.RUTA_PARTICIONES)
    validar_entrada(df)

    print(f"[info] {cfg.RESOLUCION_ORIGINAL}x{cfg.RESOLUCION_ORIGINAL} RGB -> "
          f"{cfg.RESOLUCION}x{cfg.RESOLUCION} grises, normalizado a [0, 1]")
    tensores = construir_tensores(df)
    validar_salida(tensores)

    cfg.RUTA_TENSORES.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cfg.RUTA_TENSORES, clases=np.array(cfg.CLASES), **tensores)
    peso = cfg.RUTA_TENSORES.stat().st_size / 1024**2
    print(f"[guardado] {cfg.RUTA_TENSORES.name} -> {peso:.0f} MB")


main()
