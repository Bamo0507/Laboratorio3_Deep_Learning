# Codebook: `02_tensores.npz`

Documenta el dataset final que consumen los modelos. Se genera con el pipeline
`01_particiones.py` y luego `02_preprocesamiento.py` a partir del crudo en
`data/raw/asl_alphabet_train/`.

**Fuente:** [ASL Alphabet, Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet).
87,000 fotografías a color de 200x200 px, 29 clases (A-Z, `del`, `nothing`, `space`),
3,000 imágenes por clase.

---

## Contenido del archivo

| Arreglo | Forma | Tipo | Descripción |
|---|---|---|---|
| `x_train` | (12180, 64, 64, 1) | float32 | Imágenes de entrenamiento, grises, normalizadas a [0, 1] |
| `y_train` | (12180,) | int16 | Índice de clase, posición dentro de `clases` |
| `x_val` | (2610, 64, 64, 1) | float32 | Imágenes de validación |
| `y_val` | (2610,) | int16 | Etiquetas de validación |
| `x_test` | (2610, 64, 64, 1) | float32 | Imágenes de prueba |
| `y_test` | (2610,) | int16 | Etiquetas de prueba |
| `clases` | (29,) | str | Orden canónico de las clases, define el mapeo de índice a letra |

---

## Decisiones de preprocesamiento

### Submuestreo

El dataset completo son 87,000 imágenes, lo cual resulta pesado para el tiempo de un
laboratorio, y es por esto que se trabajó con una submuestra de 600 imágenes por clase,
para un total de 17,400. Cabe mencionar que el enunciado sugiere un rango de 500 a 800
por clase, de tal forma que el valor elegido queda dentro de lo recomendado y aún deja
margen para subirlo si el tiempo de entrenamiento lo permite. La selección se hizo con
una semilla fija, de tal forma que los tres integrantes del grupo obtenemos exactamente
la misma submuestra al correr el pipeline y los modelos siguen siendo comparables entre sí.

### Conversión a escala de grises

Se decidió trabajar en escala de grises y no en RGB. La razón principal es que, por la
naturaleza del problema, una seña del alfabeto ASL se define por la **forma** que toma la
mano, es decir, por la posición de los dedos, la orientación de la palma y la posición
del pulgar, y no por el color. Por ejemplo, la diferencia entre M, N y S está únicamente en
cuántos dedos cubren al pulgar, y esa distinción se conserva íntegra en un canal de
luminancia. Bajo esta idea, los dos canales adicionales de color no aportan información
discriminante para la tarea y sí triplican el tamaño del tensor de entrada y la cantidad de
parámetros de la primera capa convolucional.

No obstante, vale la pena aclarar un punto que a primera vista parecería una objeción:
podría pensarse que al pasar a grises se elimina el tono de piel del análisis y que con eso
se pierde la posibilidad de discutir el sesgo del dataset. Considero que no es así. La
escala de grises no borra el tono de piel, lo reexpresa como luminancia, ya que una persona
de piel morena y una de piel clara siguen produciendo valores de píxel sistemáticamente
distintos dentro de la misma escala. 

### Cambio de resolución

Las imágenes se reescalaron de 200x200 a 64x64 utilizando el filtro LANCZOS, que es el
recomendado para reducción de tamaño porque promedia sobre una vecindad amplia y de esta
forma preserva mejor los bordes que un muestreo simple. La reducción lleva cada imagen de
40,000 a 4,096 píxeles, es decir, cerca de un décimo del costo original.

Cabe mencionar que esta decisión no es gratuita. Dado que los grupos visualmente similares
se distinguen por detalles finos, como la posición del pulgar en M, N y S, existe el riesgo
de que a 64x64 esos detalles se pierdan y el modelo confunda esas clases.

### Normalización

Finalmente, los píxeles se dividieron entre 255 para llevarlos del rango [0, 255] al rango
[0, 1]. Esto se hace para poder mantener las magnitudes de entrada pequeñas y homogéneas,
ya que valores grandes producen gradientes grandes en las primeras capas y eso vuelve el
entrenamiento inestable y más lento de converger.

---

## Política de valores faltantes

No aplica imputación en este dataset. Las 17,400 imágenes se validan al construir los
tensores y el pipeline verifica de forma explícita que no existan `NaN` y que todos los
píxeles queden dentro de [0, 1]; si alguna imagen no cumple, el script se detiene en vez de
arrastrar el dato malo hacia el entrenamiento.

---
