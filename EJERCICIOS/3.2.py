# Capitulo 3 - Reto Personal: Contador automatico de objetos
import cv2
import numpy as np

def detectar_formas(contorno):
    peri = cv2.arcLength(contorno, True)
    aproximacion = cv2.approxPolyDP(contorno, 0.04 * peri, True)
    vertices = len(aproximacion)
    
    if vertices == 3: return "Triangulo"
    elif vertices == 4:
        (x, y, w, h) = cv2.boundingRect(aproximacion)
        aspect_ratio = w / float(h)
        if 0.95 <= aspect_ratio <= 1.05: return "Cuadrado"
        else: return "Rectangulo"
    elif vertices > 6:
        area = cv2.contourArea(contorno)
        (x, y), radio = cv2.minEnclosingCircle(contorno)
        area_circulo = np.pi * radio ** 2
        if abs(area - area_circulo) / area_circulo < 0.3: return "Circulo"
    return "Desconocido"

def main():
    cap = cv2.VideoCapture(0)
    objetos_detectados = set()
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        desenfoque = cv2.GaussianBlur(gris, (5, 5), 0)
        bordes = cv2.Canny(desenfoque, 50, 150)
        contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        contadores = {"Triangulo": 0, "Cuadrado": 0, "Rectangulo": 0, "Circulo": 0}
        
        for contorno in contornos:
            area = cv2.contourArea(contorno)
            if area < 500: continue
            
            forma = detectar_formas(contorno)
            if forma in contadores:
                contadores[forma] += 1
                x, y, w, h = cv2.boundingRect(contorno)
                
                centro_x = x + w // 2
                centro_y = y + h // 2
                pos_aprox = (centro_x // 50 * 50, centro_y // 50 * 50, forma)
                
                if pos_aprox not in objetos_detectados:
                    print('\a')
                    objetos_detectados.add(pos_aprox)
                
                cv2.drawContours(frame, [contorno], -1, (0, 255, 0), 2)
                cv2.putText(frame, forma, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        texto_contador = f"Triangulos: {contadores['Triangulo']} | Cuadrados: {contadores['Cuadrado']} | Circulos: {contadores['Circulo']}"
        cv2.putText(frame, texto_contador, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.imshow('Detector de Formas', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()