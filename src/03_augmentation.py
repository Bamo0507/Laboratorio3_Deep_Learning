"""
Etapa 3 del pipeline: AUMENTO DE DATOS
------------------------------------------------------------
Una sola responsabilidad: expandir el conjunto de entrenamiento con copias transformadas.

  Lee : data/processed/02_tensores.npz
  Hace: - genera FACTOR_AUMENTO copias de cada imagen de entrenamiento
        - aplica rotacion, zoom, desplazamiento, brillo y contraste
        - concatena originales y copias, recorta los pixeles a [0, 1]
  Escribe: data/processed/03_tensores_aumentados.npz

Solo se aumenta ENTRENAMIENTO. Validacion y prueba se copian tal cual, porque
transformarlas cambiaria las condiciones de evaluacion y los resultados dejarian
de ser comparables con los modelos entrenados sin aumento.

No se aplica ningun tipo de espejo. Una seña esta hecha con una mano especifica y su
imagen espejo corresponde a la otra mano, de forma que un flip no aumenta los datos
sino que introduce una pose que no existe en el dataset.

Ejecutar:  python src/03_augmentation.py
"""

import numpy as np
import tensorflow as tf

import config as cfg
from utils import afirmar, banner

PARTICIONES = ["train", "val", "test"]
LOTE = 512


def construir_transformador():
    """Cadena de transformaciones aleatorias, sin espejos."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomRotation(cfg.ROTACION_GRADOS / 360, fill_mode="nearest"),
        tf.keras.layers.RandomZoom(cfg.ZOOM_MAX, fill_mode="nearest"),
        tf.keras.layers.RandomTranslation(cfg.DESPLAZAMIENTO_MAX, cfg.DESPLAZAMIENTO_MAX,
                                          fill_mode="nearest"),
        tf.keras.layers.RandomBrightness(cfg.BRILLO_MAX, value_range=(0.0, 1.0)),
        tf.keras.layers.RandomContrast(cfg.CONTRASTE_MAX),
    ], name="aumento")


def transformar(x, transformador):
    """Aplica la cadena por lotes para no cargar todo el tensor de una sola vez."""
    salida = np.zeros_like(x)
    for i in range(0, len(x), LOTE):
        trozo = tf.convert_to_tensor(x[i:i + LOTE])
        salida[i:i + LOTE] = transformador(trozo, training=True).numpy()
    return np.clip(salida, 0.0, 1.0)


def validar_entrada(datos):
    """Contrato de entrada: lo que prometio la etapa 2."""
    for p in PARTICIONES:
        afirmar(f"x_{p}" in datos and f"y_{p}" in datos, f"existen los tensores de {p}")
    afirmar(datos["x_train"].shape[1:] == cfg.FORMA_ENTRADA, f"entrada {cfg.FORMA_ENTRADA}")


def validar_salida(original, aumentado):
    """Contrato de salida: creció solo entrenamiento y los rangos siguen sanos."""
    esperado = len(original["x_train"]) * (1 + cfg.FACTOR_AUMENTO)
    afirmar(len(aumentado["x_train"]) == esperado, f"entrenamiento paso a {esperado} imagenes")
    afirmar(len(aumentado["y_train"]) == esperado, "hay una etiqueta por imagen")

    for p in ["val", "test"]:
        afirmar(np.array_equal(aumentado[f"x_{p}"], original[f"x_{p}"]), f"{p} quedo intacto")

    x = aumentado["x_train"]
    afirmar(x.min() >= 0.0 and x.max() <= 1.0, f"pixeles en [{x.min():.2f}, {x.max():.2f}]")
    afirmar(not np.isnan(x).any(), "no hay NaN")

    conteo = np.bincount(aumentado["y_train"], minlength=cfg.NUM_CLASES)
    afirmar(len(set(conteo)) == 1, f"las {cfg.NUM_CLASES} clases quedaron balanceadas ({conteo[0]} c/u)")


def main():
    banner("etapa 3: aumento de datos")

    datos = np.load(cfg.RUTA_TENSORES, allow_pickle=False)
    validar_entrada(datos)
    print(f"[cargado]  02_tensores.npz -> train {len(datos['x_train'])}")

    print(f"[info] rotacion +/-{cfg.ROTACION_GRADOS} grados, zoom +/-{cfg.ZOOM_MAX:.0%}, "
          f"desplazamiento +/-{cfg.DESPLAZAMIENTO_MAX:.0%}, "
          f"brillo +/-{cfg.BRILLO_MAX:.0%}, contraste +/-{cfg.CONTRASTE_MAX:.0%}")
    print("[info] sin espejos: un flip cambiaria la mano con la que se hace la seña")

    tf.keras.utils.set_random_seed(cfg.SEMILLA)
    transformador = construir_transformador()

    copias_x = [datos["x_train"]]
    copias_y = [datos["y_train"]]
    for n in range(cfg.FACTOR_AUMENTO):
        print(f"[info] generando copia {n + 1} de {cfg.FACTOR_AUMENTO}")
        copias_x.append(transformar(datos["x_train"], transformador))
        copias_y.append(datos["y_train"])

    aumentado = {
        "x_train": np.concatenate(copias_x).astype(np.float32),
        "y_train": np.concatenate(copias_y),
        "x_val": datos["x_val"], "y_val": datos["y_val"],
        "x_test": datos["x_test"], "y_test": datos["y_test"],
    }

    validar_salida(datos, aumentado)

    cfg.RUTA_TENSORES_AUMENTADOS.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cfg.RUTA_TENSORES_AUMENTADOS, clases=np.array(cfg.CLASES), **aumentado)
    peso = cfg.RUTA_TENSORES_AUMENTADOS.stat().st_size / 1024**2
    print(f"[guardado] {cfg.RUTA_TENSORES_AUMENTADOS.name} -> {peso:.0f} MB")


main()
