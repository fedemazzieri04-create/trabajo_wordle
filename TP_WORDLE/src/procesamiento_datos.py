import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from src import configuracion as cfg
from src import motor_logico

def palabra_a_vector_Y(palabra):
    # Transforma la palabra en un vector de 130 ceros y unos
    matriz = np.full((cfg.POSICIONES, cfg.CANTIDAD_LETRAS), cfg.TARGET_FALSO, dtype=np.int8)
    for i, letra in enumerate(palabra):
        matriz[i, ord(letra) - ord('A')] = cfg.TARGET_VERDADERO
    return matriz.flatten()

def procesar_datos():
    print("\n[PASO 1] Construyendo Datasets...")
        
    df_normal = pd.read_csv(cfg.NORMAL_CSV).assign(origen='normal')
    df_normal['id_orig'] = df_normal.index + 2
    
    df_hard = pd.read_csv(cfg.HARD_CSV).assign(origen='hard')
    df_hard['id_orig'] = df_hard.index + 2
    
    # Juntamos los dos archivos
    df_completo = pd.concat([df_normal, df_hard], ignore_index=True)
    
    # Buscamos todas las columnas de score
    score_cols = [col for col in df_completo.columns if str(col).startswith('hits_')]
    
    # Creamos una máscara booleana: verifica si alguna celda en esas columnas tiene 'GGGGG'
    mask_victorias = df_completo[score_cols].apply(lambda x: x.str.strip().str.upper() == 'GGGGG').any(axis=1)
    
    # Sobrescribimos el df conservando únicamente donde el humano ganó
    df_completo = df_completo[mask_victorias].copy()
    
    datos_procesados = []
    
    # Recorremos cada partida del archivo completo (ahora filtrado)
    for _, fila in df_completo.iterrows():
        matriz_score = motor_logico.inicializar_estado()
       
        i = 0
        continuar_partida = True
        
        while i < 4 and continuar_partida:
            col_siguiente = f"word_{i+1}"
            
            # Si ya no hay más columnas, paramos
            if col_siguiente not in df_completo.columns: 
                continuar_partida = False
            else:
                palabra = str(fila[f"word_{i}"]).strip().upper()
                score = str(fila[f"hits_{i}"]).strip().upper()
                siguiente = str(fila[col_siguiente]).strip().upper()
                
                # Si el turno está vacío, paramos
                if not siguiente or siguiente == 'NAN' or not palabra or palabra == 'NAN': 
                    continuar_partida = False
                else:
                    matriz_score = motor_logico.actualizar_estado(matriz_score, palabra, score)
                        
                    datos_procesados.append({
                        'Partida_ID': f"{fila['origen']}_{fila['id_orig']}",
                        'Turno_Actual': i,
                        'Intento_X': palabra,
                        'Score_X': matriz_score[0].flatten().tolist(),
                        'Siguiente_Y': siguiente,
                        'Score_Y': palabra_a_vector_Y(siguiente).tolist()
                    })
            
            i += 1
                
    pd.DataFrame(datos_procesados).to_csv(cfg.DATASET_FINAL_CSV, index=False)
    print(f" Dataset final (solo victorias): {cfg.DATASET_FINAL_CSV}")

    return True

def separar_train_test():

    df = pd.read_csv(cfg.DATASET_FINAL_CSV)
    df['Score_X'] = df['Score_X'].apply(ast.literal_eval) #apply(ast.literal_eval) lee el texto "[1, 0]" y te devuelve una lista [1, 0]
    df['Score_Y'] = df['Score_Y'].apply(ast.literal_eval)
    
    partidas = df['Partida_ID'].unique()
    partidas_train, partidas_test = train_test_split(partidas, test_size=0.3, random_state=cfg.RANDOM_SEED)

    df_train = df[df['Partida_ID'].isin(partidas_train)]

    pd.DataFrame(df_train).to_csv(cfg.TRAIN_CSV, index=False)
    print(f"Archivo para train: {cfg.TRAIN_CSV}")


    df_test = df[df['Partida_ID'].isin(partidas_test)]
    datos_test_limpios = []
    
    for p_id in partidas_test:
        filas = df_test[df_test['Partida_ID'] == p_id].sort_values(by='Turno_Actual')
        if len(filas) > 0:
            datos_test_limpios.append({
                'Partida_ID': p_id,
                'Palabra_Inicial': filas.iloc[0]['Intento_X'],
                'Palabra_Objetivo': filas.iloc[-1]['Siguiente_Y']
            })
            
    pd.DataFrame(datos_test_limpios).to_csv(cfg.TEST_CSV, index=False)
    print(f"Archivo para test: {cfg.TEST_CSV}")

    return True


def analizar_dataset():
    print("\nEvaluación de contradicciones y palabras con letras repetidas")

    # Cáculo de victorias humanas en el dataset original
    df_normal = pd.read_csv(cfg.NORMAL_CSV)
    df_hard = pd.read_csv(cfg.HARD_CSV)
    df_orig = pd.concat([df_normal, df_hard], ignore_index=True)
        
    score_cols = [col for col in df_orig.columns if str(col).startswith('hits_')]
    mask_victorias = df_orig[score_cols].apply(lambda x: x.str.strip().str.upper() == 'GGGGG').any(axis=1)
    win_rate = mask_victorias.mean() * 100
    print(f"\nPorcentaje de victorias humanas (dataset original): {win_rate:.2f}%\n")

    # Evaluamos salidas repetidas:

    df = pd.read_csv(cfg.DATASET_FINAL_CSV)

    agrupado = df.groupby('Score_X')['Siguiente_Y'].nunique().reset_index()
    
    estados_unicos = len(agrupado)
    unanimes = len(agrupado[agrupado['Siguiente_Y'] == 1])
    casos = agrupado[agrupado['Siguiente_Y'] > 1]
    no_unanimes = len(casos)
    # estados_unicos == unanimes + no_unanimes
    
    print(f"Estados únicos de tablero: {estados_unicos}")
    print(f"Decisiones unánimes: {unanimes}")
    print(f"Decisiones con contradicción: {no_unanimes}")

    # agarramos los 5 peores nno unanimos
    peores_casos = casos.sort_values(by='Siguiente_Y', ascending=False).head(5) 

    if not peores_casos.empty: # verificamos que no esta vacio
        opcion = input("Ver algunos casos? S/N ")
        if opcion == 'S':
            for _, caso in peores_casos.iterrows():
                estado = caso['Score_X']
                cant_variantes = caso['Siguiente_Y']

                print(f"\n MISMO ESTADO DE TABLERO DERIVÓ EN {cant_variantes} DECISIONES DISTINTAS:")
                
                filas_ejemplo = df[df['Score_X'] == estado].drop_duplicates(subset=['Siguiente_Y'])
                
                for _, fila in filas_ejemplo.iterrows():
                    partida_id = fila['Partida_ID']
                    turno = fila['Turno_Actual']
                    jugada = fila['Intento_X']
                    prediccion = fila['Siguiente_Y']
                    
                    print(f"    -> Partida {partida_id} (Turno {turno}): jugó '{jugada}' y luego tiró '{prediccion}'")
    else:
        print("\nNo hubo contradicciones.")

    # Ahora evaluamos palabras con letras repetidas:    

    print("Evaluación de palabras y letras")
    
    todas_palabras = pd.concat([df['Intento_X'], df['Siguiente_Y']]).dropna().unique()
    
    cant_repetidas = 0
    for palabra in todas_palabras:
        palabra = str(palabra)
        if len(set(palabra)) < len(palabra):
            cant_repetidas += 1
            
    total_palabras = len(todas_palabras)
    porcentaje = (cant_repetidas / total_palabras) * 100 if total_palabras > 0 else 0
    
    print(f"Palabras totales únicas en el dataset: {total_palabras}")
    print(f"Cantidad de palabras con 1 o más letras repetidas: {cant_repetidas}, lo que representa un {porcentaje:.2f}%")


    promedio_palabras = (df.groupby('Partida_ID')['Turno_Actual'].max() + 2).mean()
    print(f"\nPromedio de palabras jugadas para ganar: {promedio_palabras:.2f}")

