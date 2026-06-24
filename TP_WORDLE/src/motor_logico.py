import numpy as np
from src import configuracion as cfg

def inicializar_estado():
    matriz = np.full((cfg.POSICIONES, cfg.CANTIDAD_LETRAS), cfg.VALOR_NEUTRO, dtype=np.int8)
    limites_exactos = np.full(cfg.CANTIDAD_LETRAS, cfg.POSICIONES, dtype=np.int8)
    return (matriz, limites_exactos)

def actualizar_estado(estado_tupla, palabra, score):
    matriz, limites_exactos = estado_tupla

    nuevo_estado = np.copy(matriz)
    nuevos_limites = np.copy(limites_exactos)
    
    palabra = palabra.upper()
    score = score.upper()
    letras_unicas = set(palabra)
    
    for letra in letras_unicas:
        idx_letra = ord(letra) - ord('A') # transformación a índice numérico (0-25)
        
        # obtenemos en qué posiciones jugamos esta letra y qué colores nos devolvió el juego
        indices = [i for i, c in enumerate(palabra) if c == letra]
        score_letra = [score[i] for i in indices]
        
        tiene_gris = 'B' in score_letra
        tiene_color = 'G' in score_letra or 'Y' in score_letra
        existe_previamente = np.any(nuevo_estado[:, idx_letra] > 0)
        
        # cantidad de apariciones confirmadas en la palabra (Verdes + Amarillos)
        colores = sum(1 for h in score_letra if h in ['G', 'Y'])
        
        # Actualización de restricciones de límite
        if tiene_gris:
            nuevos_limites[idx_letra] = colores
            
        # CASO 1: letra 100% gris 
        if tiene_gris and not tiene_color:
            if existe_previamente:
                # si esta en otras posiciones, la negamos (-1) en la posicion actual
                for i in indices: nuevo_estado[i, idx_letra] = cfg.VALOR_IMPOSIBLE
            else:
                # si no la jugamos, la borramos del abecedario
                for j in range(cfg.POSICIONES): nuevo_estado[j, idx_letra] = cfg.VALOR_IMPOSIBLE
            continue
            
        # CASO 2: procesamiento espacial por posición jugada
        for i in indices:
            h = score[i]
            if h == 'G':
                nuevo_estado[i, idx_letra] = cfg.VALOR_CONFIRMADO
                # bloqueamos el resto del abecedario para esta posición 
                for l_idx in range(cfg.CANTIDAD_LETRAS):
                    if l_idx != idx_letra: nuevo_estado[i, l_idx] = cfg.VALOR_IMPOSIBLE
            elif h in ['Y', 'B']: 
                nuevo_estado[i, idx_letra] = cfg.VALOR_IMPOSIBLE
                
        # CASO 3: propagación de posibilidades
        if tiene_color:
            for j in range(cfg.POSICIONES):
                # marcamos como posible donde no hayamos jugado esta letra 
                if j not in indices and nuevo_estado[j, idx_letra] == cfg.VALOR_NEUTRO:
                    nuevo_estado[j, idx_letra] = cfg.VALOR_POSIBLE
                        
        # CASO 4: limpieza por límites descubiertos
        verdes_actuales = sum(1 for j in range(cfg.POSICIONES) if nuevo_estado[j, idx_letra] == cfg.VALOR_CONFIRMADO)
        
        # si la cantidad de verdes descubiertos es igual al límite exacto de la letra
        if verdes_actuales == nuevos_limites[idx_letra] and nuevos_limites[idx_letra] > 0:
            for j in range(cfg.POSICIONES):
                # todo lo que era posible pasa a ser imposible
                if nuevo_estado[j, idx_letra] != cfg.VALOR_CONFIRMADO:
                    nuevo_estado[j, idx_letra] = cfg.VALOR_IMPOSIBLE

    return (nuevo_estado, nuevos_limites)
