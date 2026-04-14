import cv2
import mediapipe as mp
import numpy as np

def main():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=1, 
        min_detection_confidence=0.7, 
        min_tracking_confidence=0.5
    )
    
    cap = cv2.VideoCapture(0)
    
    vol_bar = 400 # Posición inicial de la barra
    vol_per = 0   # Porcentaje inicial
    silenciado = False
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        resultados = hands.process(rgb)
        
        if resultados.multi_hand_landmarks:
            for hand_landmarks in resultados.multi_hand_landmarks:
                
                # Obtener puntos clave
                pulgar = hand_landmarks.landmark[4]
                indice = hand_landmarks.landmark[8]
                medio = hand_landmarks.landmark[12]
                muneca = hand_landmarks.landmark[0]
                
                x_p, y_p = int(pulgar.x * w), int(pulgar.y * h)
                x_i, y_i = int(indice.x * w), int(indice.y * h)
                x_m, y_m = int(medio.x * w), int(medio.y * h)
                x_w, y_w = int(muneca.x * w), int(muneca.y * h)
                
                # Check para silenciar (Puño: dedos índice y medio muy cerca de la muñeca)
                dist_i_w = np.linalg.norm(np.array([x_i, y_i]) - np.array([x_w, y_w]))
                dist_m_w = np.linalg.norm(np.array([x_m, y_m]) - np.array([x_w, y_w]))
                
                if dist_i_w < 60 and dist_m_w < 60:
                    silenciado = True
                    cv2.putText(frame, "SILENCIADO", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                else:
                    silenciado = False
                
                if not silenciado:
                    # Dibujar línea entre pulgar e índice
                    cv2.circle(frame, (x_p, y_p), 10, (255, 0, 255), -1)
                    cv2.circle(frame, (x_i, y_i), 10, (255, 0, 255), -1)
                    cv2.line(frame, (x_p, y_p), (x_i, y_i), (255, 0, 255), 3)
                    
                    # Calcular distancia y mapear a volumen
                    distancia = np.linalg.norm(np.array([x_p, y_p]) - np.array([x_i, y_i]))
                    
                    # np.interp convierte el rango de distancia [30, 200] al rango de la barra [400, 150] y porcentaje [0, 100]
                    vol_per = np.interp(distancia, [30, 200], [0, 100])
                    vol_bar = np.interp(distancia, [30, 200], [400, 150])
                    
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
        # Dibujar Interfaz de Volumen
        cv2.rectangle(frame, (50, 150), (85, 400), (0, 255, 0), 3) # Borde
        
        if not silenciado:
            cv2.rectangle(frame, (50, int(vol_bar)), (85, 400), (0, 255, 0), -1) # Relleno
            cv2.putText(frame, f'{int(vol_per)} %', (40, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        else:
            cv2.putText(frame, '0 %', (40, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
        cv2.imshow('Control de Volumen (Visual)', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    cap.release()
    hands.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()