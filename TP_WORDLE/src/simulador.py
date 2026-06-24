from numpy.random import random
import pandas as pd
import numpy as np
import joblib
import ast
from src import configuracion as cfg
from src import procesamiento_datos
from src import motor_logico

def calcular_score(intento, objetivo):
    score, obj_list = ['B']*5, list(objetivo)
    for i in range(5):
        if intento[i] == objetivo[i]: score[i], obj_list[i] = 'G', None
    for i in range(5):
        if score[i] == 'B' and intento[i] in obj_list: score[i], obj_list[obj_list.index(intento[i])] = 'Y', None
    return "".join(score)

def decodificar(probs, matriz_estado,turno):
    probs_reshaped = probs.reshape((cfg.POSICIONES, cfg.CANTIDAD_LETRAS))
    palabra = ""

    prob_explotacion = 1.0 
    if turno <= 2: prob_explotacion = 0.10
    elif turno <= 3: prob_explotacion = 0.40
    
    for i in range(cfg.POSICIONES):
        probs_pos = np.copy(probs_reshaped[i])
        
        # 1. REGLA GRIS
        # Aplastamos SIEMPRE las letras imposibles para que no salgan nunca
        for l_idx in range(cfg.CANTIDAD_LETRAS):
            if matriz_estado[i, l_idx] == cfg.VALOR_IMPOSIBLE:
                probs_pos[l_idx] = -999.0 
                
        # 2. REGLA VERDE
        # Si es menor a prob_explotacion, obligamos a poner la verde. Si es mayor, ignoramos la verde.
        if np.any(matriz_estado[i] == cfg.VALOR_CONFIRMADO) and np.random.random() < prob_explotacion:
            idx_letra = np.argmax(matriz_estado[i] == cfg.VALOR_CONFIRMADO)
        else:
            # dejamos que la IA elija su letra favorita (que ya está filtrada sin grises)
            idx_letra = np.argmax(probs_pos)
            
        palabra += chr(idx_letra + ord('A'))
        
    return palabra

def ver_metricas(df_train, df_test, modelo):

    df_train['Score_X'] = df_train['Score_X'].apply(ast.literal_eval) #apply(ast.literal_eval) lee el texto "[1, 0]" y te devuelve una lista [1, 0]
    df_train['Score_Y'] = df_train['Score_Y'].apply(ast.literal_eval)

    # METRICAS PARA TRAIN
    # Exactitud por palabra completa
    X_train = np.stack(df_train['Score_X'].values)
    Y_train = np.stack(df_train['Score_Y'].values)
    acc_palabras = modelo.score(X_train, Y_train) * 100
    
    # Exactitud letra por letra
    Y_pred = modelo.predict(X_train)
    
    # nos fijamos si los 1s en predichos coinciden con los 1s reales
    letras_reales = np.sum(Y_train == 1)
    letras_predichas_bien = np.sum((Y_train == 1) & (Y_pred == 1))
    acc_letras = (letras_predichas_bien / letras_reales) * 100 if letras_reales > 0 else 0
    
    print(f"\nMétricas Train:")
    print(f"Precisión de victorias en train (por palabra completa intermedias y finales): {acc_palabras:.2f}%")
    print(f"Precisión de letras en train (letra por letra correcta intermedias y finales): {acc_letras:.2f}%")


    # METRICAS PARA TEST 
    ganadas_5 = 0
    letras_correctas = 0
    intentos_exitosos = [] 
    cantidad_perdidas_15 = 0
    total_partidas = len(df_test)
    
    print(f"\nSimulando {total_partidas} partidas")
    
    for _, row in df_test.iterrows():
        actual = row['Palabra_Inicial']
        objetivo = row['Palabra_Objetivo']
        estado = motor_logico.inicializar_estado()
        
        gano = False
        pred = actual 
        
        intentos = 1
        if actual == objetivo:
            gano = True
        
        for turno in range(2, 16):  # hasta 15 turnos
            if gano:
                break
                
            score = calcular_score(actual, objetivo)
            estado = motor_logico.actualizar_estado(estado, actual, score)
            pred = decodificar(modelo.predict_proba([estado[0].flatten()])[0], estado[0], turno)
            intentos = turno
            
            if pred == objetivo:
                gano = True
                
            actual = pred
            
        if gano:
            intentos_exitosos.append(intentos)
            if intentos <= 5:
                ganadas_5 += 1
        else:
            cantidad_perdidas_15 += 1
        
        # Sumamos cuántas letras exactas logró en su ÚLTIMO intento
        letras_correctas += sum(1 for p, o in zip(pred, objetivo) if p == o)
            
    # Cálculos finales de métricas

    acc_letras = (letras_correctas / (total_partidas * 5)) * 100

    ganadas_mayor_5 = sum(1 for i in intentos_exitosos if i > 5)
    
    # Perdidas = las que pasaron de 15 + las que adivinó pero tardó más de 5 turnos
    cantidad_perdidas_5 = cantidad_perdidas_15 + ganadas_mayor_5
    
    porcentaje_ganadas_5 = (ganadas_5 / total_partidas) * 100 if total_partidas > 0 else 0
    porcentaje_perdidas_5 = (cantidad_perdidas_5 / total_partidas) * 100 if total_partidas > 0 else 0
    prom_palabras_15 = sum(intentos_exitosos) / len(intentos_exitosos) if intentos_exitosos else 0
    ganadas_15 = total_partidas - cantidad_perdidas_15
    porcentaje_ganadas_15 = (ganadas_15 / total_partidas) * 100 if total_partidas > 0 else 0
    porcentaje_perdidas_15 = (cantidad_perdidas_15 / total_partidas) * 100 if total_partidas > 0 else 0

    # Impresión final
    print(f"\nMétricas para test:")
    print(f" Numero total de partidas: {total_partidas}")
    print(f" Ganadas (hasta 5 turnos): {ganadas_5}")
    print(f" Porcentaje palabras ganadas (hasta 5 turnos): {round(porcentaje_ganadas_5, 2)}%")
    print(f" Partidas Perdidas (hasta 5 turnos): {cantidad_perdidas_5}")
    print(f" Porcentaje palabras perdidas (hasta 5 turnos): {round(porcentaje_perdidas_5, 2)}%")
    print(f" Ganadas (hasta 15 turnos): {ganadas_15}")
    print(f" Porcentaje palabras ganadas (hasta 15 turnos): {round(porcentaje_ganadas_15, 2)}%")
    print(f" Extremadamente Perdidas (hasta 15 turnos): {cantidad_perdidas_15}")
    print(f" Porcentaje palabras extremadamente perdidas (hasta 15 turnos): {round(porcentaje_perdidas_15, 2)}%")
    print(f" Promedio de palabras para ganar(max 15): {round(prom_palabras_15, 2)}")
    print(f" Precisión de letras en max 15 turnos: {acc_letras:.2f}%")


def inspeccionar_partida(p_id, df_test, modelo):
    if p_id not in df_test['Partida_ID'].values:
        print(f"\nLa partida '{p_id}' NO está en el set de Test.")
        return
        
    row = df_test[df_test['Partida_ID'] == p_id].iloc[0]
    actual = row['Palabra_Inicial']
    objetivo = row['Palabra_Objetivo']
    estado = motor_logico.inicializar_estado()
    
    print("\n" + "="*50)
    print(f" SIMULACIÓN MLP | PARTIDA: {p_id} | Objetivo: {objetivo}")
    print("="*50)
    print(f"  Turno 0: {actual}")
        
    gano = False
    for turno in range(1, 5): # Permitimos 4 intentos predichos (5 en total contanto inicial)
        score = calcular_score(actual, objetivo)
        estado = motor_logico.actualizar_estado(estado, actual, score)
        pred = decodificar(modelo.predict_proba([estado[0].flatten()])[0], estado[0], turno)
        
        print(f"  Colores: {score} -> MLP predice Turno {turno}: {pred}")
        if pred == objetivo: gano = True; break
        actual = pred
        
    print("  *** MLP GANÓ ***" if gano else "  --- MLP PERDIÓ ---")
    
    # BUSCAMOS AL HUMANO ORIGINAL PARA COMPARAR
    print("\n" + "="*50)
    print(f" REALIDAD HUMANA | PARTIDA: {p_id}")
    print("="*50)
    
    origen, idx_str = p_id.split('_')
    # Restamos los 2 que sumamos antes para encontrar la fila real de pandas
    idx_pandas = int(idx_str) - 2
    ruta_csv = cfg.NORMAL_CSV if origen == 'normal' else cfg.HARD_CSV
    
    try:
        df_orig = pd.read_csv(ruta_csv)
        if 0 <= idx_pandas < len(df_orig):
            fila = df_orig.iloc[idx_pandas]
            print(f"  Turno 0: {fila['word_0']}")
            for t in range(5):
                if f"word_{t+1}" not in fila or pd.isna(fila[f"word_{t+1}"]): break
                print(f"  Colores: {fila[f'hits_{t}']} -> Humano jugó Turno {t+1}: {fila[f'word_{t+1}']}")
        else:
            print("  [Error] Índice fuera de rango.")
    except Exception as e:
        print(f"  [Error al leer el CSV original]: {e}")


def jugar_partida(palabra_inicial, palabra_objetivo, modelo):
    palabra_inicial = palabra_inicial.upper()
    palabra_objetivo = palabra_objetivo.upper()

    print()
    print(f"Intento 1: {palabra_inicial}")
    
    if palabra_inicial == palabra_objetivo:
        print("¡Se adivinó a la palabra en el primer intento!")
        return

    actual = palabra_inicial
    estado = motor_logico.inicializar_estado()

    # Iteramos para los intentos 2 al 5
    for turno in range(2, 6):
        score = calcular_score(actual, palabra_objetivo)
        estado = motor_logico.actualizar_estado(estado, actual, score)
        pred = decodificar(modelo.predict_proba([estado[0].flatten()])[0], estado[0], turno)
        
        print(f"Intento {turno}: {pred}")
        
        if pred == palabra_objetivo:
            print(f"¡Se adivinó a la palabra en {turno} intentos!")
            return
            
        actual = pred

    print(f"No se logró adivinar la palabra objetivo en 5 intentos.")


def sugerir_palabra(modelo):
    print("")
    print("Ingresa las palabras que ya jugaste y el resultado de colores.")
    print("Usa: G (Verde), Y (Amarillo), B (Gris). Ejemplo: GYBBB")
    
    estado = motor_logico.inicializar_estado()
    
    while True:
        palabra = input("\nPalabra jugada (o 'S' para salir): ").strip().upper()
        if palabra == 'S':
            break
            
        if len(palabra) != cfg.POSICIONES:
            print(f"Error: La palabra debe tener {cfg.POSICIONES} letras.")
            continue
            
        score = input("Colores: ").strip().upper()
        if len(score) != cfg.POSICIONES or not all(c in 'GYB' for c in score):
            print("Error: Formato de colores inválido.")
            continue
            
        # Actualizamos la matriz de estado con la jugada
        estado = motor_logico.actualizar_estado(estado, palabra, score)
        
        # El modelo predice en base al estado aplanado (flatten)
        probs = modelo.predict_proba([estado[0].flatten()])[0]
        sugerencia = decodificar(probs, estado[0], 0)
        
        print(f"La red neuronal sugiere jugar: {sugerencia}")


def simular():
    print("\n[PASO 3] Simulador de Juego")
    df_train = pd.read_csv(cfg.TRAIN_CSV)
    df_test = pd.read_csv(cfg.TEST_CSV)
    modelo = joblib.load(cfg.MODELO_PKL)
    
    while True:
        print("\n--- MENÚ DEL SIMULADOR ---")
        print(" 1. Ver Métricas")
        print(" 2. Inspeccionar y comparar partida específica")
        print(" 3. Jugar con una palabra objetivo")
        print(" 4. Sugerir siguiente palabra")
        print(" 5. Analizar datos")
        print(" 6. Salir")
        opcion = input("Elige una opción: ")
        
        if opcion == '1':
            ver_metricas(df_train, df_test, modelo)
        elif opcion == '2':
            pid = input("\nID de la partida: ").strip().lower()
            inspeccionar_partida(pid, df_test, modelo)
        elif opcion == '3':
            palabra_obj = input("Ingresa la palabra objetivo: ")
            palabra_ini = input("Ingresa la palabra inicial: ")
            jugar_partida(palabra_ini, palabra_obj, modelo)
        elif opcion == '4':
            sugerir_palabra(modelo)
        elif opcion == '5':
            procesamiento_datos.analizar_dataset()
        elif opcion == '6':
            break
        else:
            print("Opción no válida. Intenta de nuevo.")
