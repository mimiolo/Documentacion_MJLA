import cv2
import cv2.aruco as aruco
import numpy as np

def dibujar_cubo(img, esquinas, imgpts):
    """Dibuja las líneas del cubo proyectado"""
    imgpts = np.int32(imgpts).reshape(-1, 2)
    
    # Dibujar base (z=0) en verde
    for i, j in zip([0,1,2,3], [1,2,3,0]):
        cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (0, 255, 0), 3)
    # Dibujar columnas verticales (z=0 a z=l) en azul
    for i, j in zip([0,1,2,3], [4,5,6,7]):
        cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255, 0, 0), 3)
    # Dibujar tapa (z=l) en rojo
    for i, j in zip([4,5,6,7], [5,6,7,4]):
        cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (0, 0, 255), 3)
    return img

def main():
    cap = cv2.VideoCapture(0)
    
    # Parámetros de cámara (aproximados)
    matriz_camara = np.array([[1000, 0, 640],
                              [0, 1000, 360],
                              [0, 0, 1]], dtype=np.float32)
    dist_coefs = np.zeros((4, 1))
    
    tamanio_marcador = 0.05  # 5 cm
    lado_cubo = 0.03  # 3 cm
    
    # Vértices del cubo en 3D (sistema del marcador)
    cubo_3d = np.float32([
        [-lado_cubo/2, -lado_cubo/2, 0], [lado_cubo/2, -lado_cubo/2, 0],
        [lado_cubo/2, lado_cubo/2, 0], [-lado_cubo/2, lado_cubo/2, 0],
        [-lado_cubo/2, -lado_cubo/2, lado_cubo], [lado_cubo/2, -lado_cubo/2, lado_cubo],
        [lado_cubo/2, lado_cubo/2, lado_cubo], [-lado_cubo/2, lado_cubo/2, lado_cubo]
    ])
    
    diccionario = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parametros = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(diccionario, parametros)
    
    angulo_rotacion = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detectar marcadores
        esquinas, ids, _ = detector.detectMarkers(frame)
        
        if ids is not None:
            aruco.drawDetectedMarkers(frame, esquinas, ids)
            
            for i in range(len(ids)):
                obj_points = np.array([[-tamanio_marcador/2, tamanio_marcador/2, 0],
                                       [tamanio_marcador/2, tamanio_marcador/2, 0],
                                       [tamanio_marcador/2, -tamanio_marcador/2, 0],
                                       [-tamanio_marcador/2, -tamanio_marcador/2, 0]],
                                       dtype=np.float32)
                
                success, rvec, tvec = cv2.solvePnP(obj_points, esquinas[i][0], matriz_camara, dist_coefs)
                
                if success:
                    # Aplicar rotación extra
                    angulo_rotacion += 2
                    if angulo_rotacion >= 360:
                        angulo_rotacion = 0
                        
                    R_extra, _ = cv2.Rodrigues(np.array([0, angulo_rotacion * np.pi/180, 0]))
                    cubo_rotado = np.dot(cubo_3d, R_extra.T)
                    
                    # Proyectar cubo a 2D
                    imgpts, _ = cv2.projectPoints(cubo_rotado, rvec, tvec, matriz_camara, dist_coefs)
                    
                    # Dibujar
                    frame = dibujar_cubo(frame, esquinas[i][0], imgpts)
                    cv2.drawFrameAxes(frame, matriz_camara, dist_coefs, rvec, tvec, 0.03)
                    
        cv2.imshow('Cubo 3D AR', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()