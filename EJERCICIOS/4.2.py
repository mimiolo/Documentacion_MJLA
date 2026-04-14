# Capitulo 4 - Reto Personal: Corrector de Selfies
import cv2
import numpy as np

def main():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Corrector')
    def nada(x): pass
    cv2.createTrackbar('Intensidad', 'Corrector', 10, 50, nada)
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rostros = face_cascade.detectMultiScale(gris, 1.3, 5)
        frame_procesado = frame.copy()
        
        for (x, y, w, h) in rostros:
            roi_gris = gris[y:y+h, x:x+w]
            ojos = eye_cascade.detectMultiScale(roi_gris)
            if len(ojos) >= 2:
                ojos = sorted(ojos, key=lambda o: o[0])
                ojo_izq = ojos[0]
                ojo_der = ojos[-1]
                intensidad = cv2.getTrackbarPos('Intensidad', 'Corrector') / 100.0
                pts_origen = np.float32([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
                desplazamiento = w * intensidad
                pts_destino = np.float32([[x, y], [x + w, y], [x + desplazamiento, y + h], [x + w - desplazamiento, y + h]])
                matriz = cv2.getPerspectiveTransform(pts_origen, pts_destino)
                rostro_transformado = cv2.warpPerspective(frame[y:y+h, x:x+w], matriz, (w, h))
                frame_procesado[y:y+h, x:x+w] = rostro_transformado
                
        cv2.imshow('Corrector', frame_procesado)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()