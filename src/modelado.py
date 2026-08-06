"""
Modulo compartido de entrenamiento y evaluacion.

Fija el protocolo comun a TODOS los modelos del laboratorio para que la comparacion
final aisle el diseno del modelo y no el presupuesto de entrenamiento:

  Fijo   : datos, semilla, tope de epocas, EarlyStopping, metricas, formato de salida.
  Libre  : arquitectura, optimizador, learning rate, batch size, regularizacion.

Soporta dos familias:
  - Keras  (CNN y red densa): entrenar() + evaluar() + registrar()
  - sklearn (Random Forest) : entrenar_clasico() + evaluar() + registrar()

Los modelos de sklearn reciben la imagen aplanada de (64, 64, 1) a 4096 caracteristicas.
Cada modelo llega ya construido y configurado; este modulo no decide como se optimiza.

La evaluacion corre sobre validacion. La particion de prueba se reserva para la etapa
de comparacion entre las tres familias, al final del laboratorio.
"""

import json
import time

import joblib
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


def es_keras(modelo):
    return isinstance(modelo, tf.keras.Model)


def aplanar(x):
    """Convierte (N, 64, 64, 1) en (N, 4096) para los modelos de sklearn."""
    return x.reshape(len(x), -1)


def entrenar(modelo, datos, batch_size):
    """Entrena un modelo de Keras con el presupuesto compartido."""
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


def entrenar_clasico(modelo, datos):
    """Entrena un modelo de sklearn sobre las imagenes aplanadas."""
    fijar_semillas()

    inicio = time.perf_counter()
    modelo.fit(aplanar(datos["x_train"]), datos["y_train"])
    transcurrido = time.perf_counter() - inicio

    print(f"[info] entrenado en {transcurrido:.1f} s")
    return transcurrido


def predecir(modelo, x):
    """Devuelve el indice de clase predicho, sin importar la familia del modelo."""
    if es_keras(modelo):
        return modelo.predict(x, verbose=0).argmax(axis=1)
    return modelo.predict(aplanar(x))


def evaluar(modelo, datos, particion="val"):
    """Calcula accuracy y matriz de confusion sobre la particion indicada."""
    y = datos[f"y_{particion}"]
    predicciones = predecir(modelo, datos[f"x_{particion}"])

    exactitud = accuracy_score(y, predicciones)
    matriz = confusion_matrix(y, predicciones, labels=range(cfg.NUM_CLASES))

    print(f"[info] accuracy en {particion}: {exactitud:.4f}")
    return {"particion": particion, "accuracy": float(exactitud), "matriz_confusion": matriz.tolist()}


def registrar(nombre, modelo, resultado, transcurrido, historial=None, batch_size=None, notas=""):
    """Guarda el modelo entrenado y su json de metricas."""
    cfg.DIR_MODELOS.mkdir(parents=True, exist_ok=True)
    cfg.DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)

    if es_keras(modelo):
        ruta_modelo = cfg.DIR_MODELOS / f"{nombre}.keras"
        modelo.save(ruta_modelo)
    else:
        ruta_modelo = cfg.DIR_MODELOS / f"{nombre}.joblib"
        joblib.dump(modelo, ruta_modelo, compress=3)
    print(f"[guardado] {ruta_modelo.name}")

    registro = {
        "nombre": nombre,
        "notas": notas,
        "familia": "keras" if es_keras(modelo) else "sklearn",
        "accuracy": resultado["accuracy"],
        "particion": resultado["particion"],
        "tiempo_entrenamiento_seg": round(transcurrido, 1),
        "epocas_corridas": len(historial.history["loss"]) if historial else None,
        "batch_size": batch_size,
        "semilla": cfg.SEMILLA,
        "clases": cfg.CLASES,
        "matriz_confusion": resultado["matriz_confusion"],
        "historial": {k: [float(v) for v in vs] for k, vs in historial.history.items()} if historial else None,
    }

    ruta_json = cfg.DIR_RESULTADOS / f"{nombre}.json"
    ruta_json.write_text(json.dumps(registro, indent=2), encoding="utf-8")
    print(f"[guardado] {ruta_json.name}")

    return registro
