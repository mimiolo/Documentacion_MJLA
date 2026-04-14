import cv2
import cv2.aruco as aruco
import numpy as np

def main():
    # Diccionario estándar (asegúrate de imprimir marcadores DICT_6X6_250)
    diccionario = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parametros = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(diccionario, parametros)
    
    cap = cv2.VideoCapture(0)
    print("📖 Mostrando marcadores 0 y 1. Presiona 'q' para salir.")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        esquinas, ids, rechazados = detector.detectMarkers(frame)
        
        if ids is not None:
            # Dibujar el contorno por defecto de los marcadores
            aruco.drawDetectedMarkers(frame, esquinas, ids)
            
            for i, marker_id in enumerate(ids.flatten()):
                pts = esquinas[i][0].astype(int)
                x_min, y_min = np.min(pts, axis=0)
                
                # Lógica dependiendo del ID del marcador
                if marker_id == 0:
                    texto = "¡Hola! (Marcador 0)"
                    color = (0, 0, 255) # Rojo en BGR
                elif marker_id == 1:
                    texto = "Capitulo 1 (Marcador 1)"
                    color = (255, 0, 0) # Azul en BGR
                else:
                    texto = f"ID Desconocido: {marker_id}"
                    color = (0, 255, 0) # Verde
                    
                # Dibujar un recuadro extra y texto
                cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=4)
                cv2.putText(frame, texto, (x_min, y_min - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
        cv2.imshow('Libro AR - Basico', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()