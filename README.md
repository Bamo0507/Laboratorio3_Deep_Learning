# Laboratorio 3: Deep Learning, reconocimiento de lenguaje de señas (ASL)

El dataset es [ASL Alphabet de Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet):
87,000 fotografias a color de 200x200 px repartidas en 29 clases, las 26 letras del alfabeto mas
las clases `space`, `del` y `nothing`.

---

## Como navegar este proyecto

**Este README y los notebooks son el informe.** Cada notebook contiene el codigo, los resultados
y la discusion escrita del ejercicio que le corresponde, por lo que no se entrega un PDF aparte.
El orden de lectura recomendado es el de la numeracion.

| # | Notebook | Ejercicios | Que contiene |
|---|---|---|---|
| 01 | `notebooks/01_eda.ipynb` | 1, 2 | Analisis exploratorio: ejemplos de letras, variabilidad dentro de una misma clase, distribucion de clases, letras visualmente similares y definicion de las particiones |
| 02 | `notebooks/02_red_simple.ipynb` | 5 | Red neuronal densa (fully-connected) como linea base, con cuatro variantes |
| 03 | `notebooks/03_random_forest.ipynb` | 6 | Random Forest, justificacion de la seleccion del algoritmo, dos variantes y mapa de importancia por pixel |
| 04 | `notebooks/04_cnn.ipynb` | 4 | Tres redes convolucionales: base, regularizada y regularizada con dropout moderado |
| 05 | `notebooks/05_augmentation.ipynb` | 7 | Aumento de datos, respuesta a la pregunta del flip horizontal y reentrenamiento de las tres familias |
| 06 | `notebooks/06_senas_propias.ipynb` | 8 | Comparacion final de las tres familias sobre la particion de prueba, y evaluacion del mejor modelo con fotografias propias |
| 07 | `notebooks/07_accesibilidad.ipynb` | 9 | Reflexion sobre accesibilidad y sesgo, con recomendaciones concretas |

Documentos de apoyo en la raiz:

| Archivo | Contenido |
|---|---|
| `codebook.md` | Diccionario de datos del tensor final y justificacion de cada decision de preprocesamiento (ejercicio 3) |
| `plan_modelos.md` | Seleccion de modelos, protocolo comun de comparacion y plan de transformaciones |

**Donde esta cada cosa segun lo que busque:**

- La comparacion entre los tres algoritmos y cual es el mas acertado: notebook `06`.
- La efectividad de cada modelo por separado: notebooks `02`, `03` y `04`.
- Por que se eligio escala de grises, 64x64 y una submuestra de 600 por clase: `codebook.md`.
- Por que no se aplico flip horizontal: notebook `05`.

---

## Pipeline de datos

Los notebooks no preparan datos: consumen lo que produce el pipeline de `src/`. La transformacion
esta dividida en etapas encadenadas, una por archivo, y cada una valida lo que recibio de la
anterior antes de continuar.

```
data/raw/asl_alphabet_train/        87,000 jpg en 29 carpetas
    |
    |  01_particiones.py            submuestrea 600 por clase con semilla fija
    v                               y reparte en train/val/test de forma estratificada
data/processed/01_particiones.csv   17,400 rutas etiquetadas (este si se versiona)
    |
    |  02_preprocesamiento.py       escala de grises, 200x200 a 64x64 con LANCZOS,
    v                               normalizacion a [0, 1]
data/processed/02_tensores.npz      train 12,180 / val 2,610 / test 2,610
    |
    |  03_augmentation.py           rotacion, zoom, desplazamiento, brillo y contraste
    v                               solo sobre entrenamiento, sin espejos
data/processed/03_tensores_aumentados.npz    train 24,360, val y test intactos
```

| Etapa | Archivo | Responsabilidad | Entrada | Salida |
|---|---|---|---|---|
| 0 | `src/00_init.py` | Crear el entorno virtual e instalar dependencias | `requirements.txt` | `.venv/` |
| 1 | `src/01_particiones.py` | Decidir que imagenes se usan y a que particion pertenece cada una | `data/raw/` | `01_particiones.csv` |
| 2 | `src/02_preprocesamiento.py` | Convertir rutas en tensores listos para entrenar | `01_particiones.csv` | `02_tensores.npz` |
| 3 | `src/03_augmentation.py` | Expandir el conjunto de entrenamiento con copias transformadas | `02_tensores.npz` | `03_tensores_aumentados.npz` |

Archivos de apoyo:

| Archivo | Rol |
|---|---|
| `src/config.py` | Unica fuente de rutas, semilla, resolucion, particiones y parametros de entrenamiento |
| `src/utils.py` | Logging con prefijos consistentes, entrada/salida y validacion `fail-fast` |
| `src/modelado.py` | Protocolo de entrenamiento y evaluacion compartido por todos los modelos |
| `src/run_pipeline.py` | Orquestador: corre las etapas en orden y se detiene en la primera que falle |

### Por que un pipeline por etapas y no un solo script

Cada etapa tiene una sola responsabilidad, lo que permite leerla y depurarla por separado. Los
datasets intermedios quedan auditables, de forma que se puede inspeccionar el dato a medio
procesar. La validacion entre etapas hace que los errores exploten donde ocurren y con un mensaje
claro, en vez de arrastrarse hacia el entrenamiento. Y al estar numeradas, agregar un paso nuevo
es encadenar un archivo mas sin tocar lo que ya funciona, que es exactamente lo que ocurrio con
el aumento de datos.

Esa validacion no es decorativa. Durante la preparacion, la copia del dataset se corto a media
letra U y dejo 21 de las 29 clases sin ningun mensaje de error; la etapa 1 verifica que existan
las 29 clases con 3,000 imagenes cada una y falla de inmediato si no se cumple.

---

## Como reproducirlo

```bash
git clone git@github.com:Bamo0507/Laboratorio3_Deep_Learning.git
cd Laboratorio3_Deep_Learning

python3.12 src/00_init.py          # crea .venv e instala dependencias
source .venv/bin/activate          # en Windows: .venv\Scripts\activate
```

TensorFlow no publica wheels para Python 3.13 ni 3.14, por lo que el entorno debe crearse con
Python 3.12.

El dataset **no viene en el repositorio**: son 87,000 archivos y cerca de 1 GB. Hay que
descargarlo de Kaggle y dejarlo en `data/raw/asl_alphabet_train/`, con una carpeta por clase. El
zip de Kaggle trae las carpetas duplicadas (`asl_alphabet_train/asl_alphabet_train/`), asi que hay
que quitar ese nivel extra. Para verificar que la copia quedo completa:

```bash
ls data/raw/asl_alphabet_train | wc -l                    # debe dar 29
find data/raw/asl_alphabet_train -name "*.jpg" | wc -l    # debe dar 87000
```

Con el dataset en su lugar:

```bash
python src/run_pipeline.py         # regenera los tensores desde el crudo
```

Despues se pueden abrir los notebooks en orden, seleccionando el kernel del `.venv` del proyecto.

Todo lo generado es reproducible: la semilla esta fija en `config.py` y `01_particiones.csv` se
versiona, de forma que las tres personas del grupo entrenan y evaluan sobre exactamente las mismas
imagenes. Cabe mencionar que el entrenamiento de las redes en CPU no es completamente
determinista, ya que el paralelismo entre hilos altera el orden de las sumas en punto flotante, por
lo que reejecutar un notebook produce cifras ligeramente distintas a las documentadas.

---

## Resultados

Todas las cifras de la columna de validacion salen de `resultados/*.json`, que se versionan junto
con los modelos entrenados en `models/`.

### Redes convolucionales (ejercicio 4)

| Modelo | Accuracy val | Epocas | Tiempo | Configuracion |
|---|---|---|---|---|
| `cnn_base` | 0.9590 | 21 | 428.1 s | 3 bloques conv 32/64/64, densa 128, sin regularizacion |
| `cnn_regularizada_suave` | 0.7345 | 14 | 304.0 s | Mismo esqueleto, dropout 0.1 y 0.2 |
| `cnn_regularizada` | 0.2218 | 20 | 451.7 s | Mismo esqueleto, dropout 0.25 y 0.5, BatchNorm |

Las tres comparten el mismo esqueleto convolucional y los mismos hiperparametros, de forma que la
unica variable del experimento es la regularizacion y su intensidad. La base gana con claridad: no
mostraba sobreajuste (brecha de 0.0311 entre entrenamiento y validacion), asi que la regularizacion
atacaba un problema que no existia y solo le resto capacidad de aprendizaje.

### Red neuronal simple (ejercicio 5)

| Modelo | Accuracy val | Configuracion |
|---|---|---|
| `red_simple_v4` | 0.5812 | 2 capas ocultas 1024-512, dropout 0.4, lr 0.0001, batch 128 |
| `red_simple_v2` | 0.1989 | 1 capa oculta de 1024, dropout 0.5, lr 0.001 |
| `red_simple_v3` | 0.1322 | 2 capas ocultas 512-256, dropout 0.3, lr 0.001 |
| `red_simple_v1` | 0.1291 | 1 capa oculta de 512, dropout 0.3, lr 0.001 |

La distancia entre 0.5812 y el 0.9590 de la CNN cuantifica lo que aporta la convolucion en este
problema: al aplanar la imagen se destruye la relacion espacial entre pixeles vecinos.

### Random Forest (ejercicio 6)

| Modelo | Accuracy val | Tiempo | Configuracion |
|---|---|---|---|
| `random_forest_v2` | 0.9655 | 53.3 s | 500 arboles, sin limite de profundidad |
| `random_forest_v1` | 0.9644 | 9.7 s | 200 arboles, sin limite de profundidad |

Triplicar la cantidad de arboles gana 0.0011 de accuracy a cambio de mas de cinco veces el tiempo,
por lo que se selecciona la variante de 200.

### Aumento de datos (ejercicio 7)

| Familia | Sin aumento | Con aumento | Diferencia |
|---|---|---|---|
| CNN base | 0.9590 | 0.9640 | +0.0050 |
| Red simple | 0.5812 | 0.6080 | +0.0268 |
| Random Forest | 0.9644 | 0.9686 | +0.0042 |

Las tres familias mejoraron y ninguna empeoro. No se aplico ningun tipo de espejo: una seña esta
hecha con una mano especifica, de forma que su imagen reflejada corresponde a la otra mano y no a
una variacion de la misma seña.

### Comparacion final sobre la particion de prueba

`x_test` se mantuvo sin usar durante todo el laboratorio y se evaluo una sola vez, con las tres
variantes campeonas.

| Modelo | Accuracy val | Accuracy test |
|---|---|---|
| `random_forest_aug` | 0.9686 | **0.9709** |
| `cnn_base_aug` | 0.9640 | 0.9529 |
| `red_simple_aug` | 0.6080 | 0.6195 |

El orden es el mismo en validacion y en prueba, por lo que la seleccion del Random Forest como
mejor modelo es robusta.

---

## El hallazgo principal

El mejor modelo del laboratorio acierta el 97% de las imagenes de Kaggle y **ninguna de las 15
fotografias tomadas por el grupo**. En 10 de esos 15 casos predijo `nothing`, es decir que ni
siquiera reconocio que habia una mano en la imagen.

Ese contraste, de 0.9709 a 0.0000, es la conclusion central del trabajo: un accuracy alto sobre un
dataset controlado no garantiza absolutamente nada sobre el comportamiento en un escenario de uso
real. El dataset de Kaggle esta grabado en pocas sesiones con fondo, iluminacion y encuadre
consistentes, de forma que los modelos aprendieron tanto la forma de la mano como el entorno en el
que fue fotografiada.

El desarrollo completo esta en el notebook `06`, y el analisis de que limitaciones tiene el dataset
y que haria falta para llevar el prototipo a un producto real esta en el `07`.

---

## Estructura del repositorio

```
Laboratorio3_Deep_Learning/
  src/                    pipeline por etapas, config, utilidades y protocolo de modelado
  notebooks/              un notebook por ejercicio, con codigo, resultados y discusion
  models/                 modelos entrenados (.keras y .joblib)
  resultados/             metricas de cada modelo en json
  data/
    raw/                  dataset de Kaggle (no se versiona)
    processed/            tensores generados (no se versionan, salvo 01_particiones.csv)
    own_signs/            fotografias propias del grupo, una carpeta por integrante
  codebook.md             diccionario de datos y decisiones de preprocesamiento
  plan_modelos.md         seleccion de modelos y protocolo de comparacion
  requirements.txt
```

El criterio de versionado es que todo lo que se pueda regenerar corriendo el pipeline se queda
fuera del repositorio. La unica excepcion son las fotografias de `data/own_signs/`, que no se
pueden volver a descargar de ningun lado, y `01_particiones.csv`, que es el contrato compartido que
garantiza que los tres integrantes trabajen sobre las mismas imagenes.
