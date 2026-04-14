import cv2
import numpy as np
import os
import glob

def crear_tablero_ajedrez(tamanio_cuadro=30, num_cuadros_x=9, num_cuadros_y=6):
    """Crea un tablero de ajedrez para imprimir"""
    ancho = tamanio_cuadro * num_cuadros_x
    alto = tamanio_cuadro * num_cuadros_y
    tablero = np.ones((alto, ancho), dtype=np.uint8) * 255
    
    for i in range(num_cuadros_y):
        for j in range(num_cuadros_x):
            if (i + j) % 2 == 0:
                x1, y1 = j * tamanio_cuadro, i * tamanio_cuadro
                x2, y2 = x1 + tamanio_cuadro, y1 + tamanio_cuadro
                tablero[y1:y2, x1:x2] = 0
                
    # Añadir borde
    tablero = cv2.copyMakeBorder(tablero, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=255)
    return tablero

def capturar_para_calibracion():
    """Captura imágenes del tablero desde diferentes ángulos"""
    os.makedirs("calibracion", exist_ok=True)
    cap = cv2.VideoCapture(0)
    contador = 0
    criterios = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    patron = (8, 5)  # Esquinas internas para un tablero 9x6
    
    print("📸 Mueve el tablero frente a la cámara")
    print("Presiona 'c' para capturar, 'q' para terminar")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret_encontrado, esquinas = cv2.findChessboardCorners(gris, patron, None)
        
        if ret_encontrado:
            esquinas_subpix = cv2.cornerSubPix(gris, esquinas, (11,11), (-1,-1), criterios)
            frame_con_esquinas = cv2.drawChessboardCorners(frame.copy(), patron, esquinas_subpix, ret_encontrado)
            cv2.putText(frame_con_esquinas, "Tablero detectado! Presiona 'c'", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Calibracion', frame_con_esquinas)
        else:
            cv2.imshow('Calibracion', frame)
            
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord('c') and ret_encontrado:
            nombre = f"calibracion/img_{contador:03d}.png"
            cv2.imwrite(nombre, frame)
            print(f"✅ Capturada {nombre}")
            contador += 1
        elif tecla == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print(f"🎉 Total de {contador} imágenes capturadas")

def calibrar_camara():
    """Calcula matriz de cámara y coeficientes de distorsión"""
    patron = (8, 5)
    tamanio_cuadro = 0.025  # 2.5 cm
    puntos_3d = np.zeros((patron[0] * patron[1], 3), np.float32)
    puntos_3d[:, :2] = np.mgrid[0:patron[0], 0:patron[1]].T.reshape(-1, 2)
    puntos_3d *= tamanio_cuadro
    
    puntos_3d_lista = []
    puntos_2d_lista = []
    imagenes = glob.glob('calibracion/*.png')
    
    if len(imagenes) == 0:
        print("❌ No se encontraron imágenes en /calibracion/")
        return None, None
        
    for nombre_imagen in imagenes:
        imagen = cv2.imread(nombre_imagen)
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        ret, esquinas = cv2.findChessboardCorners(gris, patron, None)
        if ret:
            puntos_3d_lista.append(puntos_3d)
            criterios = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            esquinas_refinadas = cv2.cornerSubPix(gris, esquinas, (11,11), (-1,-1), criterios)
            puntos_2d_lista.append(esquinas_refinadas)
            
    if len(puntos_2d_lista) == 0: return None, None
    
    print("\n🔧 Calculando parámetros...")
    ret, matriz_camara, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(puntos_3d_lista, puntos_2d_lista, gris.shape[::-1], None, None)
    np.savez('parametros_camara.npz', matriz_camara=matriz_camara, dist_coefs=dist_coefs)
    print("\n💾 Parámetros guardados en 'parametros_camara.npz'")
    return matriz_camara, dist_coefs

def corregir_distorsion(imagen, matriz_camara, dist_coefs):
    h, w = imagen.shape[:2]
    nueva_matriz, roi = cv2.getOptimalNewCameraMatrix(matriz_camara, dist_coefs, (w, h), 1, (w, h))
    imagen_corregida = cv2.undistort(imagen, matriz_camara, dist_coefs, None, nueva_matriz)
    x, y, w, h = roi
    return imagen_corregida[y:y+h, x:x+w]

if __name__ == "__main__":
    # 1. Crear el tablero
    cv2.imwrite("tablero_calibracion.png", crear_tablero_ajedrez())
    print("✅ Tablero creado. Imprímelo para la captura.")
    
    # 2. Iniciar captura
    capturar_para_calibracion()
    
    # 3. Procesar calibración
    matriz, dist = calibrar_camara()
    
    # 4. Probar en vivo
    if matriz is not None:
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret: break
            corregido = corregir_distorsion(frame, matriz, dist)
            cv2.imshow('Original', frame)
            cv2.imshow('Corregido', corregido)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        cap.release()
        cv2.destroyAllWindows()