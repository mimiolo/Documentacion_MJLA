import cv2
import mediapipe as mp
import numpy as np

def main():
    mp_hands = mp.solutions.hands
    # Solo detectamos una mano para no confundir el dibujo
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    
    cap = cv2.VideoCapture(0)
    
    # Lienzo en blanco del mismo tamaño que la cámara
    lienzo = None
    x_prev, y_prev = 0, 0
    
    print("🎨 Pizarra Mágica Básica.")
    print("- Dibuja moviendo tu dedo índice.")
    print("- Presiona 'c' para limpiar el lienzo.")
    print("- Presiona 'q' para salir.")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Efecto espejo para que sea natural dibujar
        frame = cv2.flip(frame, 1)
        
        # Inicializar el lienzo la primera vez
        if lienzo is None:
            lienzo = np.zeros_like(frame)
            
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        
        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0]
            
            # Obtener posición del dedo índice (punto 8)
            h, w = frame.shape[:2]
            x, y = int(lm.landmark[8].x * w), int(lm.landmark[8].y * h)
            
            # Dibujar un círculo en la punta del dedo para saber dónde estamos
            cv2.circle(frame, (x, y), 10, (255, 0, 0), cv2.FILLED)
            
            # Lógica de dibujo
            if x_prev == 0 and y_prev == 0:
                x_prev, y_prev = x, y
                
            # Dibujar línea en el lienzo
            cv2.line(lienzo, (x_prev, y_prev), (x, y), (255, 0, 0), 5)
            
            x_prev, y_prev = x, y
        else:
            # Si no hay mano, reiniciar la posición previa
            x_prev, y_prev = 0, 0
            
        # Fusionar el frame original con el lienzo usando operaciones de bits
        frame_gris = cv2.cvtColor(lienzo, cv2.COLOR_BGR2GRAY)
        _, mascara = cv2.threshold(frame_gris, 50, 255, cv2.THRESH_BINARY_INV)
        frame_fondo = cv2.bitwise_and(frame, frame, mask=mascara)
        frame_final = cv2.bitwise_or(frame_fondo, lienzo)
        
        cv2.imshow('Pizarra Virtual', frame_final)
        
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord('q'): 
            break
        elif tecla == ord('c'):
            lienzo = np.zeros_like(frame) # Limpiar
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()