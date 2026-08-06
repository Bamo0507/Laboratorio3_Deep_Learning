"""Constantes globales del laboratorio. Unica fuente de rutas y parametros."""

from pathlib import Path

# ------------------------------------------------------------
# Rutas
# ------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent.parent
DIR_RAW = RAIZ / "data" / "raw"
DIR_PROCESSED = RAIZ / "data" / "processed"
DIR_FIGURAS = RAIZ / "data" / "processed" / "figuras"

RUTA_TRAIN_CRUDO = DIR_RAW / "asl_alphabet_train"
RUTA_TEST_CRUDO = DIR_RAW / "asl_alphabet_test"
RUTA_SENAS_PROPIAS = RAIZ / "data" / "own_signs"

RUTA_PARTICIONES = DIR_PROCESSED / "01_particiones.csv"
RUTA_TENSORES = DIR_PROCESSED / "02_tensores.npz"

# ------------------------------------------------------------
# Reproducibilidad
# ------------------------------------------------------------
SEMILLA = 123

# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------
CLASES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "del", "nothing", "space",
]
NUM_CLASES = len(CLASES)
IMGS_POR_CLASE_CRUDO = 3000
TOTAL_CRUDO = NUM_CLASES * IMGS_POR_CLASE_CRUDO
RESOLUCION_ORIGINAL = 200

GRUPOS_CONFUNDIBLES = {
    "puno_cerrado": ["M", "N", "S", "T", "A", "E"],
    "dedos_extendidos": ["U", "V", "R", "W"],
    "curvas": ["C", "O"],
    "meniques": ["I", "J", "Y"],
}

# ------------------------------------------------------------
# Submuestreo
# ------------------------------------------------------------
IMGS_POR_CLASE = 600
RESOLUCION = 64
RESOLUCION_ALTA = 96
CANALES = 3
FORMA_ENTRADA = (RESOLUCION, RESOLUCION, CANALES)

# ------------------------------------------------------------
# Particiones
# ------------------------------------------------------------
PROP_TRAIN = 0.70
PROP_VAL = 0.15
PROP_TEST = 0.15

# ------------------------------------------------------------
# Entrenamiento
# ------------------------------------------------------------
BATCH_SIZE = 64
EPOCAS = 20
PACIENCIA = 4
