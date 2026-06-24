import os
from src import procesamiento_datos
from src import modelo_neuronal
from src import simulador
from src import configuracion as cfg

def main():
    print("="*60)
    print(" PIPELINE DE IA PARA WORDLE ".center(60))
    print("="*60)
    
    if not os.path.exists(cfg.NORMAL_CSV) or not os.path.exists(cfg.HARD_CSV):
        print(f"\n[!] ALERTA: No se encontraron los archivos CSV originales.")
        print(f"Por favor, coloca 'normal.csv' y 'hard.csv' dentro de la carpeta: datos/")
        return

    if not os.path.exists(cfg.DATASET_FINAL_CSV):
        exito = procesamiento_datos.procesar_datos()
        if not exito: return
        exito = procesamiento_datos.separar_train_test()
        if not exito: return
    else:
        print("\n[PASO 1] Los datasets ya existen. Omitiendo generación...")
    
    if not os.path.exists(cfg.MODELO_PKL):
        modelo_neuronal.entrenar_modelo()
    else:
        print("\n[PASO 2] El modelo ya está entrenado. Omitiendo entrenamiento...")
        
    simulador.simular()
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
