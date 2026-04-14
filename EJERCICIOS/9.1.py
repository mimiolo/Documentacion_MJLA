import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QGroupBox, QListWidget)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap

class DetectorMarcadores(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Detector de Marcadores ArUco - Capítulo 9")
        self.setGeometry(100, 100, 1000, 600)
        
        # Configuración moderna de ArUco (OpenCV 4.7+)
        diccionario = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        parametros = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(diccionario, parametros)
        
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
        
        grupo_info = QGroupBox("📌 Marcadores Detectados")
        layout_info = QVBoxLayout()
        
        self.lista_marcadores = QListWidget()
        self.lista_marcadores.setStyleSheet("font-size: 16px;")
        layout_info.addWidget(self.lista_marcadores)
        
        self.label_total = QLabel("Total en pantalla: 0")
        self.label_total.setStyleSheet("font-size: 14px; font-weight: bold; color: #4CAF50;")
        layout_info.addWidget(self.label_total)
        
        grupo_info.setLayout(layout_info)
        layout_control.addWidget(grupo_info)
        
        layout_control.addStretch()
        layout.addWidget(panel_control, 1)

    def actualizar_frame(self):
        ret, frame = self.cap.read()
        if not ret: return
        
        # Detección de marcadores
        esquinas, ids, rechazados = self.detector.detectMarkers(frame)
        
        # Limpiar la lista para actualizarla
        self.lista_marcadores.clear()
        
        if ids is not None:
            # Dibujar los contornos y los IDs sobre la imagen
            cv2.aruco.drawDetectedMarkers(frame, esquinas, ids)
            
            # Actualizar la interfaz
            self.label_total.setText(f"Total en pantalla: {len(ids)}")
            for i in range(len(ids)):
                id_marcador = int(ids[i][0])
                self.lista_marcadores.addItem(f"Marcador ID: {id_marcador}")
        else:
            self.label_total.setText("Total en pantalla: 0")
            
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
    ventana = DetectorMarcadores()
    ventana.show()
    sys.exit(app.exec())