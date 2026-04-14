import cv2
import numpy as np
import sys

def mostrar_valor_pixel(evento, x, y, flags, param):
    """Callback que muestra el valor del píxel bajo el cursor"""
    if evento == cv2.EVENT_MOUSEMOVE:
        imagen = param
        # Verificar que el cursor esté dentro de los límites de la imagen
        if 0 <= y < imagen.shape[0] and 0 <= x < imagen.shape[1]:
            bgr = imagen[y, x]
            
            # Crear una copia para no dibujar permanentemente sobre la original
            img_info = imagen.copy()
            
            # Formatear el texto con los valores BGR
            texto = f"B:{bgr[0]:3d} G:{bgr[1]:3d} R:{bgr[2]:3d}"
            
            # Dibujar el texto y un círculo indicador
            cv2.putText(img_info, texto, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.circle(img_info, (x, y), 5, (0, 255, 0), 2)
            
            cv2.imshow('Visor', img_info)

def main():
    # Paso 1: Cargar imagen (busca 'test.jpg' en tu carpeta)
    imagen = cv2.imread('test.jpg')
    
    if imagen is None:
        # Si no existe, creamos una imagen de gradiente automática para que el código no falle
        imagen = np.zeros((400, 600, 3), dtype=np.uint8)
        for i in range(400):
            for j in range(600):
                imagen[i, j] = [j % 256, i % 256, (i+j) % 256]
        print("⚠️ Imagen 'test.jpg' no encontrada. Generando imagen de prueba...")

    # Mostrar información técnica en la terminal
    print(f"📊 Dimensiones de la imagen (Alto, Ancho, Canales): {imagen.shape}")
    print(f"📦 Tipo de dato: {imagen.dtype}")
    print(f"🔢 Total de píxeles: {imagen.size}")

    # Configurar la ventana y el evento del mouse
    cv2.namedWindow('Visor')
    cv2.setMouseCallback('Visor', mostrar_valor_pixel, imagen)

    print("\n💡 Instrucciones: Mueve el mouse sobre la imagen. Presiona 'q' o 'ESC' para salir.")

    while True:
        cv2.imshow('Visor', imagen)
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord('q') or tecla == 27:  # q o ESC para salir
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()