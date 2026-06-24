import os
import numpy as np
import pandas as pd
import joblib
import ast
import copy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", message="The y_prob values do not sum to one.")
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from src import configuracion as cfg
from src import motor_logico
from src import simulador

def evaluar_partidas_val(modelo, df_val):
    ganadas_5 = 0
    intentos_ganados = []
    
    # Agrupamos por Partida_ID para jugar juegos enteros sin mezclar
    partidas_agrupadas = df_val.groupby('Partida_ID')
    total_partidas = len(partidas_agrupadas)
    
    for _, partida in partidas_agrupadas:
        partida = partida.sort_values('Turno_Actual')
        # Asumimos que la primera palabra de Intento_X es el inicio, y la última de Siguiente_Y es el objetivo
        palabra_actual = partida.iloc[0]['Intento_X']
        objetivo = partida.iloc[-1]['Siguiente_Y'] 
        
        estado_actual = motor_logico.inicializar_estado()
        gano = False
        
        # Eliminamos el break usando la variable gano
        turno = 1
        while turno <= 15 and not gano:  # hasta 15 turnos
            if palabra_actual == objetivo:
                gano = True
                intentos_ganados.append(turno)
                if turno <= 5:
                    ganadas_5 += 1
            else:
                score = simulador.calcular_score(palabra_actual, objetivo)
                estado_actual = motor_logico.actualizar_estado(estado_actual, palabra_actual, score)
                pred_proba = modelo.predict_proba([estado_actual[0].flatten()])[0]
                palabra_actual = simulador.decodificar(pred_proba, estado_actual[0], turno)
                turno += 1
            
    # Partidas perdidas son las que no están en intentos_ganados
    partidas_ganadas_totales = len(intentos_ganados)
    cantidad_perdidas_15 = total_partidas - partidas_ganadas_totales
    
    # Calculamos las perdidas mayores a 5 pero que si adivinaron antes de 15
    ganadas_mayor_5 = 0
    for intentos in intentos_ganados:
        if intentos > 5:
             ganadas_mayor_5 += 1
             
    # Las partidas perdidas (menor a 5 turnos) son las perdidas totales más las que adivinaron en más de 5 turnos
    cantidad_perdidas_5 = cantidad_perdidas_15 + ganadas_mayor_5

    promedio_palabras_15 = np.mean(intentos_ganados) if intentos_ganados else 0

    porcentaje_ganadas_5 = (ganadas_5 / total_partidas) * 100 if total_partidas > 0 else 0
    porcentaje_perdidas_5 = (cantidad_perdidas_5 / total_partidas) * 100 if total_partidas > 0 else 0

    return ganadas_5, porcentaje_ganadas_5, promedio_palabras_15, cantidad_perdidas_15, cantidad_perdidas_5, porcentaje_perdidas_5, total_partidas

def graficar_loss_train_val(train_loss, val_loss, nombre_modelo):
    plt.figure(figsize=(8, 5))
    plt.plot(train_loss, label='Train Loss', color='blue')
    plt.plot(val_loss, label='Val Loss', color='orange')
    plt.title(f'Loss vs Épocas - {nombre_modelo}')
    plt.xlabel('Épocas')
    plt.ylabel('Loss (Cross Entropy)')
    plt.legend()
    plt.grid(alpha=0.5)
    plt.savefig(f"{cfg.RESULTADOS}loss_{nombre_modelo}.png")
    plt.close()

def entrenar_modelo():
    print("\n[PASO 2] Entrenar MLPs y comparar resultados")
    os.makedirs(cfg.RESULTADOS, exist_ok=True)

    df = pd.read_csv(cfg.TRAIN_CSV)
    df['Score_X'] = df['Score_X'].apply(ast.literal_eval)
    df['Score_Y'] = df['Score_Y'].apply(ast.literal_eval)

    partidas_unicas = df['Partida_ID'].unique()
    partidas_train, partidas_val = train_test_split(partidas_unicas, test_size=0.2, random_state=cfg.RANDOM_SEED)

    df_train = df[df['Partida_ID'].isin(partidas_train)]
    df_val = df[df['Partida_ID'].isin(partidas_val)]

    X_train = np.stack(df_train['Score_X'].values)
    Y_train = np.stack(df_train['Score_Y'].values)
    X_val = np.stack(df_val['Score_X'].values)
    Y_val = np.stack(df_val['Score_Y'].values)
    
    # modelos
    modelos = {
        "MLP_base": MLPClassifier(hidden_layer_sizes=(256, 128), activation='relu', learning_rate='adaptive', warm_start=True, max_iter=1, random_state=cfg.RANDOM_SEED),
        "MLP_intermedio": MLPClassifier(hidden_layer_sizes=(512, 256), activation='relu', learning_rate='adaptive', warm_start=True, max_iter=1, random_state=cfg.RANDOM_SEED),
        "MLP_profundo": MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation='relu', learning_rate='adaptive', warm_start=True, max_iter=1, random_state=cfg.RANDOM_SEED),
    }

    resultados = []
    epocas_totales = 300

    mejor_modelo = None
    max_ganadas = -1
    nombre_ganador = ""

    for nombre, modelo in modelos.items():
        train_loss, val_loss = [], []
        mejor_loss_val = float('inf')
        mejor_modelo_estado = None
        
        for _ in range(epocas_totales):
            modelo.fit(X_train, Y_train)
            train_loss.append(modelo.loss_)
            
            # Calculamos validación usando log_loss sobre las probabilidades
            Y_pred_val = modelo.predict_proba(X_val)
            loss_v = log_loss(Y_val, Y_pred_val)
            val_loss.append(loss_v)

            # guardar el modelo cuando minimiza la los de validacion
            if loss_v < mejor_loss_val:
                mejor_loss_val = loss_v
                mejor_modelo_estado = copy.deepcopy(modelo)
            
        modelo = mejor_modelo_estado
        # guardar modelo y graficar
        joblib.dump(modelo, f"{cfg.RESULTADOS}{nombre}.pkl")
        graficar_loss_train_val(train_loss, val_loss, nombre)
        
        # métricas finales sobre el dataset de validación
        ganadas_5, porcentaje_ganadas_5, prom_palabras_15, cantidad_perdidas_15, cantidad_perdidas_5, porcentaje_perdidas_5, total_partidas = evaluar_partidas_val(modelo, df_val)
        
        resultados.append({
            "Modelo": nombre,
            "Total_Partidas": total_partidas,
            "Ganadas_<=5": ganadas_5,
            "Porcentaje_Ganadas_<=5": round(porcentaje_ganadas_5, 2),
            "Partidas_Perdidas_(>5)": cantidad_perdidas_5,
            "Porcentaje_Perdidas_(>5)": round(porcentaje_perdidas_5, 2),
            "Promedio_Palabras(max 15)": round(prom_palabras_15, 2),
            "Extremadamente_Perdidas_(>15)": cantidad_perdidas_15,
            "Ganadas_<=15": total_partidas - cantidad_perdidas_15
        })

        if ganadas_5 > max_ganadas:
            max_ganadas = ganadas_5
            mejor_modelo = modelo
            nombre_ganador = nombre

    # comparativo
    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_csv(f"{cfg.RESULTADOS}comparacion_modelos.csv", index=False)
    
    # Guardamos EL MEJOR MODELO en la ruta principal para que el resto del sistema lo use
    joblib.dump(mejor_modelo, cfg.MODELO_PKL)
    print(f"\nModelo seleccionado: {nombre_ganador} con {max_ganadas} victorias.")
    