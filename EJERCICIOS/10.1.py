import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QGroupBox, QFileDialog, QMessageBox)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap

class ProyectorImagenesAR(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Proyector de Imágenes AR - Capítulo 10")
        self.setGeometry(100, 100, 1100, 700)
        
        # Configurar ArUco moderno
        self.diccionario = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.parametros = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.diccionario, self.parametros)
        
        self.imagen_proyectar = None
        
        self.setup_ui()
        
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.actualizar_frame)
        self.timer.start(30)

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        
        self.label_video = QLabel()
        self.label_video.setMinimumSize(640, 480)
        self.label_video.setStyleSheet("border: 2px solid #333; background-color: #111;")
        self.label_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.label_video, 3)
        
        panel_control = QWidget()
        panel_control.setMaximumWidth(300)
        layout_control = QVBoxLayout(panel_control)
        
        grupo_controles = QGroupBox("Cargar Elemento AR")
        layout_controles = QVBoxLayout()
        
        btn_cargar = QPushButton("🖼️ Seleccionar Imagen a Proyectar")
        btn_cargar.clicked.connect(self.cargar_imagen)
        layout_controles.addWidget(btn_cargar)
        
        self.info_img = QLabel("Ninguna imagen cargada.\n(Se mostrará un cuadro de color por defecto)")
        self.info_img.setWordWrap(True)
        layout_controles.addWidget(self.info_img)
        
        grupo_controles.setLayout(layout_controles)
        layout_control.addWidget(grupo_controles)
        
        self.label_estado = QLabel("Buscando marcadores...")
        self.label_estado.setStyleSheet("font-weight: bold; color: #4CAF50;")
        layout_control.addWidget(self.label_estado)
        
        layout_control.addStretch()
        layout.addWidget(panel_control, 1)

    def cargar_imagen(self):
        archivo, _ = QFileDialog.getOpenFileName(self, "Seleccionar Imagen", "", "Imágenes (*.png *.jpg *.jpeg)")
        if archivo:
            self.imagen_proyectar = cv2.imread(archivo)
            self.info_img.setText(f"✅ Imagen cargada:\n{archivo.split('/')[-1]}")

    def superponer_imagen(self, frame, esquinas_marcador):
        # Si no hay imagen cargada, creamos un cuadrado azul por defecto
        if self.imagen_proyectar is None:
            self.imagen_proyectar = np.zeros((300, 300, 3), dtype=np.uint8)
            self.imagen_proyectar[:] = (255, 100, 0) # Color azul
            cv2.putText(self.imagen_proyectar, "CARGA UNA", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(self.imagen_proyectar, "IMAGEN", (80, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        alto_img, ancho_img = self.imagen_proyectar.shape[:2]
        alto_frame, ancho_frame = frame.shape[:2]

        # Puntos de origen (Esquinas de la imagen que cargamos)
        pts_origen = np.array([
            [0, 0], [ancho_img, 0], 
            [ancho_img, alto_img], [0, alto_img]
        ], dtype=np.float32)

        # Puntos de destino (Esquinas del marcador en la cámara)
        # ArUco devuelve las esquinas en orden: sup-izq, sup-der, inf-der, inf-izq
        pts_destino = esquinas_marcador[0].reshape(4, 2)

        # 1. Calcular la matriz de Homografía (la deformación de perspectiva)
        H, _ = cv2.findHomography(pts_origen, pts_destino)
        
        # Si H es válido, aplicamos la transformación
        if H is not None:
            # 2. Deformar la imagen
            imagen_deformada = cv2.warpPerspective(self.imagen_proyectar, H, (ancho_frame, alto_frame))
            
            # 3. Crear una máscara para hacer un "hueco" en el frame original
            mascara = np.zeros((alto_frame, ancho_frame), dtype=np.uint8)
            cv2.fillConvexPoly(mascara, np.int32(pts_destino), 255)
            mascara_invertida = cv2.bitwise_not(mascara)
            
            # 4. Juntar todo
            fondo = cv2.bitwise_and(frame, frame, mask=mascara_invertida)
            frame_final = cv2.add(fondo, imagen_deformada)
            return frame_final
            
        return frame

    def actualizar_frame(self):
        ret, frame = self.cap.read()
        if not ret: return
        
        esquinas, ids, _ = self.detector.detectMarkers(frame)
        
        if ids is not None:
            self.label_estado.setText(f"Marcadores detectados: {len(ids)}")
            # Iteramos sobre todos los marcadores detectados
            for i in range(len(ids)):
                frame = self.superponer_imagen(frame, esquinas[i])
        else:
            self.label_estado.setText("Buscando marcadores...")
            
        # Mostrar en la interfaz
        rgb_disp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_disp.shape
        qt_image = QImage(rgb_disp.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.label_video.setPixmap(QPixmap.fromImage(qt_image).scaled(self.label_video.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ventana = ProyectorImagenesAR()
    ventana.show()
    sys.exit(app.exec())