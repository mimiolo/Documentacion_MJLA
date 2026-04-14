import cv2
import mediapipe as mp
import math

def calcular_distancia(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def main():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
    cap = cv2.VideoCapture(0)
    
    PUNTO_SUP = 159
    PUNTO_INF = 145
    PUNTO_IZQ = 33
    PUNTO_DER = 133
    
    contador_parpadeos = 0
    ojo_cerrado = False
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultados = face_mesh.process(rgb)
        
        if resultados.multi_face_landmarks:
            landmarks = resultados.multi_face_landmarks[0].landmark
            
            p_sup = (int(landmarks[PUNTO_SUP].x * w), int(landmarks[PUNTO_SUP].y * h))
            p_inf = (int(landmarks[PUNTO_INF].x * w), int(landmarks[PUNTO_INF].y * h))
            p_izq = (int(landmarks[PUNTO_IZQ].x * w), int(landmarks[PUNTO_IZQ].y * h))
            p_der = (int(landmarks[PUNTO_DER].x * w), int(landmarks[PUNTO_DER].y * h))
            
            cv2.circle(frame, p_sup, 2, (0, 255, 0), -1)
            cv2.circle(frame, p_inf, 2, (0, 255, 0), -1)
            
            distancia_vertical = calcular_distancia(p_sup, p_inf)
            distancia_horizontal = calcular_distancia(p_izq, p_der)
            
            ratio = distancia_vertical / distancia_horizontal if distancia_horizontal != 0 else 0
            
            if ratio < 0.22:
                if not ojo_cerrado:
                    ojo_cerrado = True
                    contador_parpadeos += 1
            else:
                ojo_cerrado = False
                
            color_texto = (0, 0, 255) if ojo_cerrado else (255, 255, 255)
            cv2.putText(frame, f"Parpadeos: {contador_parpadeos}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color_texto, 2)
            cv2.putText(frame, f"Ratio: {ratio:.2f}", (20, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                        
        cv2.imshow("Detector de Parpadeo", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    face_mesh.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()