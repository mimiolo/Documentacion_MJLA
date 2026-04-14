import cv2
import mediapipe as mp
import random
import math

def main():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    
    cap = cv2.VideoCapture(0)
    
    # Variables del juego
    puntos = 0
    ancho_pantalla, alto_pantalla = 640, 480
    
    # Objeto que cae
    obj_x = random.randint(50, ancho_pantalla - 50)
    obj_y = 0
    obj_radio = 25
    obj_velocidad = 7
    
    print("🎮 AR Catcher Básico. ¡Usa tu mano para atrapar el círculo rojo!")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (ancho_pantalla, alto_pantalla))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        
        # Mover objeto
        obj_y += obj_velocidad
        
        # Si el objeto toca el suelo, reaparece sin puntos
        if obj_y > alto_pantalla:
            obj_y = 0
            obj_x = random.randint(50, ancho_pantalla - 50)
            
        pos_mano = None
        # Detectar mano
        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0]
            # Usar la palma/muñeca (punto 0) o el dedo índice (punto 8)
            x_mano = int(lm.landmark[8].x * ancho_pantalla)
            y_mano = int(lm.landmark[8].y * alto_pantalla)
            pos_mano = (x_mano, y_mano)
            
            # Dibujar un círculo verde en el dedo
            cv2.circle(frame, pos_mano, 20, (0, 255, 0), 3)
            
        # Lógica de colisión
        if pos_mano:
            dist = math.hypot(pos_mano[0] - obj_x, pos_mano[1] - obj_y)
            if dist < (obj_radio + 20):  # Radio del objeto + radio de la mano
                puntos += 10
                obj_velocidad += 0.5 # Aumenta la dificultad
                # Reaparecer objeto
                obj_y = 0
                obj_x = random.randint(50, ancho_pantalla - 50)
                
        # Dibujar objeto y UI
        cv2.circle(frame, (obj_x, int(obj_y)), obj_radio, (0, 0, 255), -1)
        cv2.putText(frame, f"Puntuacion: {puntos}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        cv2.imshow('Mini Juego AR', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()