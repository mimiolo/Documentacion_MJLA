import cv2
import numpy as np

def dibujar_cubo_3d(imagen, puntos_imagen):
    """ Dibuja un cubo 3D usando las coordenadas proyectadas """
    puntos_imagen = np.int32(puntos_imagen).reshape(-1, 2)
    
    # Dibujar pilares (aristas verticales)
    for i, j in zip(range(4), range(4, 8)):
        cv2.line(imagen, tuple(puntos_imagen[i]), tuple(puntos_imagen[j]), (255, 150, 0), 3)
        
    # Dibujar la tapa superior del cubo
    cv2.drawContours(imagen, [puntos_imagen[4:8]], -1, (0, 255, 0), 3)

def main():
    # Configurar ArUco
    diccionario = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parametros = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(diccionario, parametros)
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        h, w = frame.shape[:2]
        
        # 1. Aproximar los parámetros de la cámara (Matriz de intrínsecos)
        # Esto es necesario para simular el 3D sin haber calibrado la cámara
        focal_length = w
        center = (w / 2, h / 2)
        matriz_camara = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float32)
        dist_coeffs = np.zeros((4, 1)) # Sin distorsión del lente
        
        # 2. Detectar marcadores
        esquinas, ids, _ = detector.detectMarkers(frame)
        
        if ids is not None:
            # Tamaño real aproximado del marcador impreso (ej. 5 cm)
            longitud_marcador = 0.05 
            
            # Puntos 3D del marcador en el mundo real (Z=0, plano plano)
            obj_points = np.array([
                [-longitud_marcador/2,  longitud_marcador/2, 0],
                [ longitud_marcador/2,  longitud_marcador/2, 0],
                [ longitud_marcador/2, -longitud_marcador/2, 0],
                [-longitud_marcador/2, -longitud_marcador/2, 0]
            ], dtype=np.float32)
            
            # Puntos 3D que formarán nuestro cubo hacia "arriba" (Z negativo)
            eje_z = -longitud_marcador # Altura del cubo
            cubo_3d = np.float32([
                [-longitud_marcador/2,  longitud_marcador/2, 0],
                [ longitud_marcador/2,  longitud_marcador/2, 0],
                [ longitud_marcador/2, -longitud_marcador/2, 0],
                [-longitud_marcador/2, -longitud_marcador/2, 0],
                [-longitud_marcador/2,  longitud_marcador/2, eje_z],
                [ longitud_marcador/2,  longitud_marcador/2, eje_z],
                [ longitud_marcador/2, -longitud_marcador/2, eje_z],
                [-longitud_marcador/2, -longitud_marcador/2, eje_z]
            ])
            
            for i in range(len(ids)):
                # 3. Calcular la pose (rotación y traslación) del marcador respecto a la cámara
                exito, rvec, tvec = cv2.solvePnP(obj_points, esquinas[i][0], matriz_camara, dist_coeffs)
                
                if exito:
                    # 4. Proyectar los puntos 3D del cubo en la imagen 2D
                    puntos_proyectados, _ = cv2.projectPoints(cubo_3d, rvec, tvec, matriz_camara, dist_coeffs)
                    
                    # 5. Dibujar el marcador base y el cubo encima
                    cv2.aruco.drawDetectedMarkers(frame, esquinas)
                    dibujar_cubo_3d(frame, puntos_proyectados)
                    
                    cv2.putText(frame, f"ID: {ids[i][0]} (3D)", (int(esquinas[i][0][0][0]), int(esquinas[i][0][0][1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.putText(frame, "Cubo 3D en Realidad Aumentada", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
        cv2.imshow("Reto - Cubo 3D Virtual", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()