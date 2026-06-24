# src/configuracion.py

# Archivos de datos
NORMAL_CSV = "data/normal.csv"
HARD_CSV = "data/hard.csv"
DATASET_FINAL_CSV = "data/dataset_wordle.csv"
TRAIN_CSV = "data/train_partidas.csv"
TEST_CSV = "data/test_partidas.csv"

# Archivo del modelo
MODELO_PKL = "models/modelo_mlp.pkl"

RESULTADOS = "results/"

# Constantes
POSICIONES = 5
CANTIDAD_LETRAS = 26

# Entradas X
VALOR_NEUTRO = 0
VALOR_IMPOSIBLE = -1
VALOR_POSIBLE = 1
VALOR_CONFIRMADO = 2

# Salidas Y
TARGET_FALSO = 0
TARGET_VERDADERO = 1

# Semilla
RANDOM_SEED = 42
