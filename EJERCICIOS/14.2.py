import sys
import cv2
import cv2.aruco as aruco
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QTextEdit, QFrame)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap, QFont

class MotorLibroAR:
    def __init__(self):
        self.diccionario = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        self.parametros = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.diccionario, self.parametros)
        
        # Base de datos del libro
        self.paginas = {
            "0": {
                "titulo": "🌟 Portada",
                "texto": "¡Bienvenido al Libro Mágico AR!\n\nEste libro cobra vida cuando miras sus páginas a través de la cámara.\n\nPasa al marcador #1 para comenzar.",
                "color_borde": (0, 255, 255) # Amarillo
            },
            "1": {
                "titulo": "🚀 Capítulo 1: El Descubrimiento",
                "texto": "En el año 2050, la Realidad Aumentada ya no requería gafas especiales.\n\nTodo comenzó cuando un joven ingeniero encontró un viejo código en Python...",
                "color_borde": (255, 100, 100) # Azul claro
            },
            "2": {
                "titulo": "🐉 Capítulo 2: La Bestia de Píxeles",
                "texto": "¡Cuidado! Los errores de compilación tomaron forma física.\n\nSolo ordenando correctamente las matrices de NumPy se podía derrotar a la bestia.",
                "color_borde": (100, 100, 255) # Rojo claro
            }
        }

    def procesar(self, frame):
        esquinas, ids, _ = self.detector.detectMarkers(frame)
        pagina_activa = None
        
        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                id_str = str(marker_id)
                if id_str in self.paginas:
                    info = self.paginas[id_str]
                    pagina_activa = info # Guardamos la información para la UI
                    
                    # Dibujar efectos AR en el frame
                    pts = esquinas[i][0].astype(int)
                    x_min, y_min = np.min(pts, axis=0)
                    cv2.polylines(frame, [pts], True, info["color_borde"], 4)
                    cv2.putText(frame, info["titulo"], (x_min, y_min - 10), 
                                cv2.FONT_HERSHEY_DUPLEX, 0.7, info["color_borde"], 2)
                    
        return frame, pagina_activa

class LibroARApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("📖 Lector de Libro AR - Capítulo 14")
        self.setGeometry(100, 100, 1200, 700)
        
        self.motor_ar = MotorLibroAR()
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.actualizar_frame)
        self.timer.start(30)
        
        self.setup_ui()

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        
        # Panel de Cámara
        self.label_video = QLabel()
        self.label_video.setMinimumSize(800, 600)
        self.label_video.setStyleSheet("background-color: #000; border: 2px solid #555;")
        self.label_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.label_video, 2)
        
        # Panel del Libro (Texto)
        panel_libro = QFrame()
        panel_libro.setStyleSheet("background-color: #f4eecd; border-radius: 10px; border: 1px solid #dcd3a1;")
        layout_libro = QVBoxLayout(panel_libro)
        
        self.lbl_titulo = QLabel("Esperando página...")
        self.lbl_titulo.setFont(QFont("Georgia", 16, QFont.Weight.Bold))
        self.lbl_titulo.setStyleSheet("color: #4a3b32;")
        self.lbl_titulo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout_libro.addWidget(self.lbl_titulo)
        
        self.txt_contenido = QTextEdit()
        self.txt_contenido.setFont(QFont("Georgia", 14))
        self.txt_contenido.setStyleSheet("background-color: transparent; border: none; color: #333;")
        self.txt_contenido.setReadOnly(True)
        self.txt_contenido.setText("Muestra un marcador ArUco (0, 1 o 2) a la cámara para leer el contenido del libro.")
        layout_libro.addWidget(self.txt_contenido)
        
        layout.addWidget(panel_libro, 1)

    def actualizar_frame(self):
        ret, frame = self.cap.read()
        if not ret: return
        
        frame, info_pagina = self.motor_ar.procesar(frame)
        
        # Actualizar la UI del libro si detectamos una página
        if info_pagina:
            self.lbl_titulo.setText(info_pagina["titulo"])
            self.txt_contenido.setText(info_pagina["texto"])
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qt_img = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[2]*rgb.shape[1], QImage.Format.Format_RGB888)
        self.label_video.setPixmap(QPixmap.fromImage(qt_img).scaled(self.label_video.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ventana = LibroARApp()
    ventana.show()
    sys.exit(app.exec())