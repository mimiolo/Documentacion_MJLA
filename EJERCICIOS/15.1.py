import cv2
import mediapipe as mp
import numpy as np
import math

def crear_gafas_rgba():
    """Crea una imagen RGBA de unas gafas simples"""
    gafas = np.zeros((100, 250, 4), dtype=np.uint8)
    # Lentes oscuros con transparencia
    cv2.rectangle(gafas, (20, 20), (100, 70), (0, 0, 0, 200), -1)
    cv2.rectangle(gafas, (150, 20), (230, 70), (0, 0, 0, 200), -1)
    # Puente
    cv2.rectangle(gafas, (100, 30), (150, 45), (0, 0, 0, 255), -1)
    return gafas

def superponer_rgba(fondo, overlay, x, y):
    """Superpone una imagen con transparencia en coordenadas x, y"""
    h, w = overlay.shape[:2]
    # Evitar desbordamiento de pantalla
    if y < 0 or x < 0 or y + h > fondo.shape[0] or x + w > fondo.shape[1]: 
        return fondo
        
    alpha = overlay[:, :, 3] / 255.0
    for c in range(3):
        fondo[y:y+h, x:x+w, c] = fondo[y:y+h, x:x+w, c] * (1 - alpha) + overlay[:, :, c] * alpha
    return fondo

def main():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
    gafas_img = crear_gafas_rgba()
    
    cap = cv2.VideoCapture(0)
    print("👗 Probador Virtual Básico. Presiona 'q' para salir.")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Modo espejo
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            # Coordenadas de los ojos (índices 33 y 362 en FaceMesh)
            ojo_izq = (int(lm[33].x * w), int(lm[33].y * h))
            ojo_der = (int(lm[362].x * w), int(lm[362].y * h))
            
            # Centro entre los ojos
            cx = (ojo_izq[0] + ojo_der[0]) // 2
            cy = (ojo_izq[1] + ojo_der[1]) // 2
            
            # Distancia para escalar las gafas
            distancia_ojos = math.hypot(ojo_der[0] - ojo_izq[0], ojo_der[1] - ojo_izq[1])
            escala = distancia_ojos / (gafas_img.shape[1] * 0.4) # Factor de ajuste
            
            nuevo_ancho = int(gafas_img.shape[1] * escala)
            nuevo_alto = int(gafas_img.shape[0] * escala)
            
            if nuevo_ancho > 0 and nuevo_alto > 0:
                gafas_redimensionadas = cv2.resize(gafas_img, (nuevo_ancho, nuevo_alto))
                # Ajustar posición (x, y) superior izquierda
                pos_x = cx - nuevo_ancho // 2
                pos_y = cy - nuevo_alto // 2
                
                frame = superponer_rgba(frame, gafas_redimensionadas, pos_x, pos_y)
                
        cv2.imshow('Probador Basico', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()