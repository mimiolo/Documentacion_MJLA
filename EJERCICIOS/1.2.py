
import cv2
import numpy as np

def main():
    imagen = cv2.imread('test.jpg')
    if imagen is None:
        imagen = np.zeros((400, 600, 3), dtype=np.uint8)
        for i in range(400):
            for j in range(600):
                imagen[i, j] = [j % 256, i % 256, (i+j) % 256]
                
    cv2.namedWindow('Visor')
    canal_actual = 'Todos'
    
    while True:
        img_mostrar = imagen.copy()
        
        if canal_actual == 'Azul':
            img_mostrar[:, :, 1] = 0
            img_mostrar[:, :, 2] = 0
        elif canal_actual == 'Verde':
            img_mostrar[:, :, 0] = 0
            img_mostrar[:, :, 2] = 0
        elif canal_actual == 'Rojo':
            img_mostrar[:, :, 0] = 0
            img_mostrar[:, :, 1] = 0
            
        cv2.putText(img_mostrar, f"Canal: {canal_actual}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img_mostrar, "Teclas: 1(Azul) 2(Verde) 3(Rojo) 4(Todos) q(Salir)", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                    
        cv2.imshow('Visor', img_mostrar)
        
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord('1'):
            canal_actual = 'Azul'
        elif tecla == ord('2'):
            canal_actual = 'Verde'
        elif tecla == ord('3'):
            canal_actual = 'Rojo'
        elif tecla == ord('4'):
            canal_actual = 'Todos'
        elif tecla == ord('q') or tecla == 27:
            break
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()