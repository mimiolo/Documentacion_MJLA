import cv2
import mediapipe as mp

def main():
    # Inicializar MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1, 
        refine_landmarks=True, 
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    )
    
    cap = cv2.VideoCapture(0)
    print("🎭 Mostrando detección facial. Presiona 'q' para salir.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convertir a RGB para MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultados = face_mesh.process(rgb)
        
        if resultados.multi_face_landmarks:
            for face_landmarks in resultados.multi_face_landmarks:
                # Obtener dimensiones de la imagen
                h, w, _ = frame.shape
                
                # El punto 1 en MediaPipe suele ser la punta de la nariz
                nariz_x = int(face_landmarks.landmark[1].x * w)
                nariz_y = int(face_landmarks.landmark[1].y * h)
                
                # Dibujar una "nariz de payaso" roja
                cv2.circle(frame, (nariz_x, nariz_y), 20, (0, 0, 255), -1)
                
                # Opcional: Dibujar la malla completa suavemente
                # mp.solutions.drawing_utils.draw_landmarks(
                #    frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                #    landmark_drawing_spec=None,
                #    connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style())
                
        cv2.imshow('Filtro Basico - Ejercicio', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()