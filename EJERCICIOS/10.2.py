import cv2
import numpy as np

def main():
    # 1. Configurar ArUco
    diccionario = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parametros = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(diccionario, parametros)
    
    # 2. Configurar la cámara y el video a proyectar
    cap = cv2.VideoCapture(0)
    
    # PON AQUÍ EL NOMBRE DE TU VIDEO
    ruta_video = "video_ar.mp4" 
    cap_video = cv2.VideoCapture(ruta_video)
    
    if not cap_video.isOpened():
        print(f"⚠️ No se pudo cargar '{ruta_video}'.")
        print("Crea o descarga un video con ese nombre en esta carpeta.")
        return

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Leer el frame del video que vamos a superponer
        ret_video, frame_video = cap_video.read()
        
        # Si el video termina, lo reiniciamos (Bucle infinito)
        if not ret_video:
            cap_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret_video, frame_video = cap_video.read()
            
        esquinas, ids, _ = detector.detectMarkers(frame)
        
        if ids is not None:
            alto_frame, ancho_frame = frame.shape[:2]
            alto_vid, ancho_vid = frame_video.shape[:2]
            
            # Puntos del video original
            pts_origen = np.array([
                [0, 0], [ancho_vid, 0], 
                [ancho_vid, alto_vid], [0, alto_vid]
            ], dtype=np.float32)
            
            for i in range(len(ids)):
                # Puntos del marcador en el mundo real
                pts_destino = esquinas[i][0].reshape(4, 2)
                
                # Calcular Homografía
                H, _ = cv2.findHomography(pts_origen, pts_destino)
                
                if H is not None:
                    # Deformar el frame del video
                    video_deformado = cv2.warpPerspective(frame_video, H, (ancho_frame, alto_frame))
                    
                    # Máscara para borrar la parte del marcador
                    mascara = np.zeros((alto_frame, ancho_frame), dtype=np.uint8)
                    cv2.fillConvexPoly(mascara, np.int32(pts_destino), 255)
                    mascara_invertida = cv2.bitwise_not(mascara)
                    
                    # Combinar el fondo con el video deformado
                    fondo = cv2.bitwise_and(frame, frame, mask=mascara_invertida)
                    frame = cv2.add(fondo, video_deformado)

        cv2.putText(frame, "AR Video Player (Mueve el marcador)", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Reto - Reproductor AR", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    cap.release()
    cap_video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()