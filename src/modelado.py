"""
Modulo compartido de entrenamiento y evaluacion.

Fija el protocolo comun a TODOS los modelos del laboratorio para que la comparacion
final aisle la arquitectura y no el presupuesto de entrenamiento:

  Fijo   : datos, semilla, tope de epocas, EarlyStopping, metricas, formato de salida.
  Libre  : arquitectura, optimizador, learning rate, batch size, regularizacion.

Cada modelo llega ya compilado; este modulo no decide como se optimiza.
La evaluacion corre sobre validacion. La particion de prueba se reserva para la
etapa de comparacion entre las tres familias de modelos.
"""

import json
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix

import config as cfg
from utils import afirmar


def fijar_semillas():
    """Deja numpy y tensorflow en un estado reproducible."""
    tf.keras.utils.set_random_seed(cfg.SEMILLA)


def cargar_tensores():
    """Lee 02_tensores.npz y devuelve los seis arreglos mas el orden de clases."""
    datos = np.load(cfg.RUTA_TENSORES, allow_pickle=False)
    afirmar(datos["x_train"].shape[1:] == cfg.FORMA_ENTRADA, f"entrada {cfg.FORMA_ENTRADA}")
    print(f"[cargado]  tensores -> train {len(datos['x_train'])}, "
          f"val {len(datos['x_val'])}, test {len(datos['x_test'])}")
    return datos


def entrenar(modelo, datos, batch_size):
    """Entrena con el presupuesto compartido y devuelve el historial y el tiempo en segundos."""
    fijar_semillas()

    parada = tf.keras.callbacks.EarlyStopping(
        monitor=cfg.METRICA_SELECCION,
        patience=cfg.PACIENCIA,
        restore_best_weights=True,
        mode="max",
    )

    inicio = time.perf_counter()
    historial = modelo.fit(
        datos["x_train"], datos["y_train"],
        validation_data=(datos["x_val"], datos["y_val"]),
        epochs=cfg.MAX_EPOCAS,
        batch_size=batch_size,
        callbacks=[parada],
        verbose=2,
    )
    transcurrido = time.perf_counter() - inicio

    print(f"[info] {len(historial.history['loss'])} epocas en {transcurrido:.1f} s")
    return historial, transcurrido


def evaluar(modelo, datos, particion="val"):
    """Calcula accuracy y matriz de confusion sobre la particion indicada."""
    x = datos[f"x_{particion}"]
    y = datos[f"y_{particion}"]

    predicciones = modelo.predict(x, verbose=0).argmax(axis=1)
    exactitud = accuracy_score(y, predicciones)
    matriz = confusion_matrix(y, predicciones, labels=range(cfg.NUM_CLASES))

    print(f"[info] accuracy en {particion}: {exactitud:.4f}")
    return {"particion": particion, "accuracy": float(exactitud), "matriz_confusion": matriz.tolist()}


def registrar(nombre, modelo, historial, transcurrido, resultado, batch_size, notas=""):
    """Guarda el modelo entrenado y su json de metricas."""
    cfg.DIR_MODELOS.mkdir(parents=True, exist_ok=True)
    cfg.DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)

    ruta_modelo = cfg.DIR_MODELOS / f"{nombre}.keras"
    modelo.save(ruta_modelo)
    print(f"[guardado] {ruta_modelo.name}")

    registro = {
        "nombre": nombre,
        "notas": notas,
        "accuracy": resultado["accuracy"],
        "particion": resultado["particion"],
        "tiempo_entrenamiento_seg": round(transcurrido, 1),
        "epocas_corridas": len(historial.history["loss"]),
        "batch_size": batch_size,
        "max_epocas": cfg.MAX_EPOCAS,
        "semilla": cfg.SEMILLA,
        "clases": cfg.CLASES,
        "matriz_confusion": resultado["matriz_confusion"],
        "historial": {k: [float(v) for v in vs] for k, vs in historial.history.items()},
    }

    ruta_json = cfg.DIR_RESULTADOS / f"{nombre}.json"
    ruta_json.write_text(json.dumps(registro, indent=2), encoding="utf-8")
    print(f"[guardado] {ruta_json.name}")

    return registro
