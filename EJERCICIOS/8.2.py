import cv2
import mediapipe as mp
import numpy as np
import time

def calcular_angulo(a, b, c, landmarks, w, h):
    punto_a = np.array([landmarks[a].x * w, landmarks[a].y * h])
    punto_b = np.array([landmarks[b].x * w, landmarks[b].y * h])
    punto_c = np.array([landmarks[c].x * w, landmarks[c].y * h])
    
    ba = punto_a - punto_b
    bc = punto_c - punto_b
    
    cos_angulo = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angulo = np.degrees(np.arccos(np.clip(cos_angulo, -1.0, 1.0)))
    return angulo, tuple(punto_a.astype(int)), tuple(punto_b.astype(int)), tuple(punto_c.astype(int))

def main():
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    cap = cv2.VideoCapture(0)
    
    tiempo_encorvado = 0
    encorvado_inicio = None
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultados = pose.process(rgb)
        
        if resultados.pose_landmarks:
            # Hombro (11 o 12), Cadera (23 o 24), Rodilla (25 o 26)
            # Usamos el lado izquierdo por defecto
            angulo, hombro, cadera, rodilla = calcular_angulo(11, 23, 25, resultados.pose_landmarks.landmark, w, h)
            
            # Dibujar lineas de la espalda
            cv2.line(frame, hombro, cadera, (255, 0, 255), 4)
            cv2.line(frame, cadera, rodilla, (255, 0, 255), 4)
            cv2.circle(frame, hombro, 6, (0, 255, 255), -1)
            cv2.circle(frame, cadera, 6, (0, 255, 255), -1)
            cv2.circle(frame, rodilla, 6, (0, 255, 255), -1)
            
            # Postura recta es cercana a 180 grados. 
            # (Este umbral puede variar según si la cámara te ve de lado o de frente)
            if angulo < 155:
                if encorvado_inicio is None:
                    encorvado_inicio = time.time()
                tiempo_encorvado = time.time() - encorvado_inicio
                
                # Si pasas más de 5 segundos encorvado
                if tiempo_encorvado > 5:
                    cv2.putText(frame, "¡SIENTATE DERECHO!", (50, 80), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                    print('\a') # Beep de alerta en la terminal
            else:
                encorvado_inicio = None
                tiempo_encorvado = 0
                cv2.putText(frame, "Postura Correcta", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
            cv2.putText(frame, f"Angulo espalda: {angulo:.1f} grados", (10, h - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
        cv2.imshow('Detector de Postura', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    cap.release()
    pose.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()