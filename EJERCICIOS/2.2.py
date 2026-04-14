# Capitulo 2 - Reto Personal: Detector de ropa favorita
import cv2
import numpy as np
import os

def nada(x):
    pass

def main():
    cv2.namedWindow('Control')
    cv2.namedWindow('Resultado')
    
    cv2.createTrackbar('H Min', 'Control', 0, 179, nada)
    cv2.createTrackbar('H Max', 'Control', 179, 179, nada)
    cv2.createTrackbar('S Min', 'Control', 0, 255, nada)
    cv2.createTrackbar('S Max', 'Control', 255, 255, nada)
    cv2.createTrackbar('V Min', 'Control', 0, 255, nada)
    cv2.createTrackbar('V Max', 'Control', 255, 255, nada)
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        h_min = cv2.getTrackbarPos('H Min', 'Control')
        h_max = cv2.getTrackbarPos('H Max', 'Control')
        s_min = cv2.getTrackbarPos('S Min', 'Control')
        s_max = cv2.getTrackbarPos('S Max', 'Control')
        v_min = cv2.getTrackbarPos('V Min', 'Control')
        v_max = cv2.getTrackbarPos('V Max', 'Control')
        
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        
        mascara = cv2.inRange(hsv, lower, upper)
        resultado = cv2.bitwise_and(frame, frame, mask=mascara)
        
        cv2.putText(resultado, "g: Guardar | c: Cargar | r: Reset | q: Salir", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
        cv2.imshow('Resultado', resultado)
        cv2.imshow('Mascara', mascara)
        
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord('g'):
            np.save('color_favorito.npy', np.array([h_min, h_max, s_min, s_max, v_min, v_max]))
            print("Color guardado")
        elif tecla == ord('c'):
            if os.path.exists('color_favorito.npy'):
                valores = np.load('color_favorito.npy')
                cv2.setTrackbarPos('H Min', 'Control', valores[0])
                cv2.setTrackbarPos('H Max', 'Control', valores[1])
                cv2.setTrackbarPos('S Min', 'Control', valores[2])
                cv2.setTrackbarPos('S Max', 'Control', valores[3])
                cv2.setTrackbarPos('V Min', 'Control', valores[4])
                cv2.setTrackbarPos('V Max', 'Control', valores[5])
                print("Color cargado")
        elif tecla == ord('r'):
            cv2.setTrackbarPos('H Min', 'Control', 0)
            cv2.setTrackbarPos('H Max', 'Control', 179)
            cv2.setTrackbarPos('S Min', 'Control', 0)
            cv2.setTrackbarPos('S Max', 'Control', 255)
            cv2.setTrackbarPos('V Min', 'Control', 0)
            cv2.setTrackbarPos('V Max', 'Control', 255)
            print("Reset")
        elif tecla == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()