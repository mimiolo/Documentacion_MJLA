import cv2
import mediapipe as mp
import math

def calcular_angulo(p1, p2, p3):
    """Calcula el ángulo (en grados) formado por 3 puntos."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    
    # Trigonometría básica (math.atan2)
    angulo = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    
    # Asegurar que el ángulo esté entre 0 y 180
    angulo = abs(angulo)
    if angulo > 180.0:
        angulo = 360.0 - angulo
    return int(angulo)

def main():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    cap = cv2.VideoCapture(0)
    print("💪 Medidor de Flexión de Brazo. Muestra tu brazo a la cámara.")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            
            # Índices: Hombro (11), Codo (13), Muñeca (15) para brazo izquierdo
            hombro = (int(lm[11].x * w), int(lm[11].y * h))
            codo = (int(lm[13].x * w), int(lm[13].y * h))
            muneca = (int(lm[15].x * w), int(lm[15].y * h))
            
            # Dibujar líneas conectando el brazo
            cv2.line(frame, hombro, codo, (255, 255, 255), 3)
            cv2.line(frame, codo, muneca, (255, 255, 255), 3)
            
            # Dibujar puntos
            for pt in [hombro, codo, muneca]:
                cv2.circle(frame, pt, 8, (0, 0, 255), cv2.FILLED)
                
            # Calcular ángulo y mostrarlo
            angulo = calcular_angulo(hombro, codo, muneca)
            
            # Imprimir el ángulo al lado del codo
            cv2.putText(frame, f"{angulo} deg", (codo[0] + 20, codo[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
            # Feedback visual simple
            if angulo < 50:
                cv2.putText(frame, "FLEXIONADO", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            elif angulo > 150:
                cv2.putText(frame, "EXTENDIDO", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                
        cv2.imshow('Analisis de Pose', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()