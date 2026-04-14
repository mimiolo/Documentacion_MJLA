import cv2
import numpy as np

def main():
    # Usar el detector HOG integrado en OpenCV (sin internet ni descargas)
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret: 
            break
            
        # Redimensionar para que el procesamiento sea más rápido
        frame = cv2.resize(frame, (640, 480))
        
        # Detectar personas
        cajas, pesos = hog.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.05)
        
        # Dibujar rectángulos
        for (x, y, w, h) in cajas:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Persona", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
        cv2.putText(frame, f"Personas detectadas: {len(cajas)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
        cv2.imshow('Contador de Personas HOG', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()