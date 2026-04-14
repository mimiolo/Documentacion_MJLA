# Capitulo 2 - Mini Proyecto: Selector de Color Magico
import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QSlider, QPushButton, 
                             QGroupBox, QComboBox)
from PyQt6.QtCore import Qt, QTimer, QDateTime
from PyQt6.QtGui import QImage, QPixmap

class SelectorColorMagico(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Selector de Color Magico - Capitulo 2")
        self.setGeometry(100, 100, 1300, 800)
        self.cap = None
        self.camara_activa = False
        self.h_min, self.h_max = 0, 179
        self.s_min, self.s_max = 0, 255
        self.v_min, self.v_max = 0, 255
        self.colores_preset = {
            "Personalizado": (0, 179, 0, 255, 0, 255),
            "Rojo": (0, 10, 100, 255, 100, 255),
            "Rojo (alternativo)": (170, 179, 100, 255, 100, 255),
            "Verde": (40, 80, 100, 255, 100, 255),
            "Azul": (100, 130, 100, 255, 100, 255),
            "Amarillo": (20, 30, 100, 255, 100, 255),
            "Naranja": (5, 15, 100, 255, 100, 255),
            "Rosa": (140, 160, 100, 255, 100, 255),
        }
        self.setup_ui()
        self.setup_camara()
        
    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        
        panel_video = QWidget()
        layout_video = QVBoxLayout(panel_video)
        self.label_video = QLabel()
        self.label_video.setMinimumSize(800, 600)
        self.label_video.setStyleSheet("border: 2px solid #333; background-color: #111;")
        self.label_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout_video.addWidget(self.label_video)
        self.fps_label = QLabel("FPS: --")
        layout_video.addWidget(self.fps_label)
        layout.addWidget(panel_video, 3)
        
        panel_control = QWidget()
        panel_control.setMaximumWidth(400)
        layout_control = QVBoxLayout(panel_control)
        
        grupo_preset = QGroupBox("Colores Predefinidos")
        layout_preset = QVBoxLayout()
        self.combo_colores = QComboBox()
        self.combo_colores.addItems(self.colores_preset.keys())
        self.combo_colores.currentTextChanged.connect(self.cambiar_preset)
        layout_preset.addWidget(self.combo_colores)
        grupo_preset.setLayout(layout_preset)
        layout_control.addWidget(grupo_preset)
        
        grupo_hsv = QGroupBox("Control Manual HSV")
        layout_hsv = QVBoxLayout()
        
        layout_hsv.addWidget(QLabel("Hue (Matiz):"))
        slider_h_min = QSlider(Qt.Orientation.Horizontal)
        slider_h_min.setRange(0, 179)
        slider_h_min.valueChanged.connect(lambda v: self.actualizar_hsv('h_min', v))
        layout_hsv.addWidget(QLabel("  Minimo:"))
        layout_hsv.addWidget(slider_h_min)
        
        slider_h_max = QSlider(Qt.Orientation.Horizontal)
        slider_h_max.setRange(0, 179)
        slider_h_max.setValue(179)
        slider_h_max.valueChanged.connect(lambda v: self.actualizar_hsv('h_max', v))
        layout_hsv.addWidget(QLabel("  Maximo:"))
        layout_hsv.addWidget(slider_h_max)
        
        layout_hsv.addWidget(QLabel("Saturation (Saturacion):"))
        slider_s_min = QSlider(Qt.Orientation.Horizontal)
        slider_s_min.setRange(0, 255)
        slider_s_min.valueChanged.connect(lambda v: self.actualizar_hsv('s_min', v))
        layout_hsv.addWidget(QLabel("  Minimo:"))
        layout_hsv.addWidget(slider_s_min)
        
        slider_s_max = QSlider(Qt.Orientation.Horizontal)
        slider_s_max.setRange(0, 255)
        slider_s_max.setValue(255)
        slider_s_max.valueChanged.connect(lambda v: self.actualizar_hsv('s_max', v))
        layout_hsv.addWidget(QLabel("  Maximo:"))
        layout_hsv.addWidget(slider_s_max)
        
        layout_hsv.addWidget(QLabel("Value (Brillo):"))
        slider_v_min = QSlider(Qt.Orientation.Horizontal)
        slider_v_min.setRange(0, 255)
        slider_v_min.valueChanged.connect(lambda v: self.actualizar_hsv('v_min', v))
        layout_hsv.addWidget(QLabel("  Minimo:"))
        layout_hsv.addWidget(slider_v_min)
        
        slider_v_max = QSlider(Qt.Orientation.Horizontal)
        slider_v_max.setRange(0, 255)
        slider_v_max.setValue(255)
        slider_v_max.valueChanged.connect(lambda v: self.actualizar_hsv('v_max', v))
        layout_hsv.addWidget(QLabel("  Maximo:"))
        layout_hsv.addWidget(slider_v_max)
        
        grupo_hsv.setLayout(layout_hsv)
        layout_control.addWidget(grupo_hsv)
        
        grupo_info = QGroupBox("Informacion")
        layout_info = QVBoxLayout()
        self.info_rango = QLabel(f"H: [{self.h_min}, {self.h_max}]\nS: [{self.s_min}, {self.s_max}]\nV: [{self.v_min}, {self.v_max}]")
        layout_info.addWidget(self.info_rango)
        grupo_info.setLayout(layout_info)
        layout_control.addWidget(grupo_info)
        
        btn_captura = QPushButton("Guardar instantanea")
        btn_captura.clicked.connect(self.guardar_instantanea)
        layout_control.addWidget(btn_captura)
        
        layout_control.addStretch()
        layout.addWidget(panel_control, 1)
        
        self.sliders = {
            'h_min': slider_h_min, 'h_max': slider_h_max,
            's_min': slider_s_min, 's_max': slider_s_max,
            'v_min': slider_v_min, 'v_max': slider_v_max
        }
        
    def setup_camara(self):
        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened():
            self.camara_activa = True
            self.timer = QTimer()
            self.timer.timeout.connect(self.actualizar_frame)
            self.timer.start(30)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
    def actualizar_hsv(self, parametro, valor):
        setattr(self, parametro, valor)
        self.info_rango.setText(f"H: [{self.h_min}, {self.h_max}]\nS: [{self.s_min}, {self.s_max}]\nV: [{self.v_min}, {self.v_max}]")
        
    def cambiar_preset(self, nombre):
        if nombre in self.colores_preset:
            valores = self.colores_preset[nombre]
            self.h_min, self.h_max, self.s_min, self.s_max, self.v_min, self.v_max = valores
            self.sliders['h_min'].setValue(self.h_min)
            self.sliders['h_max'].setValue(self.h_max)
            self.sliders['s_min'].setValue(self.s_min)
            self.sliders['s_max'].setValue(self.s_max)
            self.sliders['v_min'].setValue(self.v_min)
            self.sliders['v_max'].setValue(self.v_max)
            
    def aplicar_efecto_cine(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([self.h_min, self.s_min, self.v_min])
        upper = np.array([self.h_max, self.s_max, self.v_max])
        mascara_color = cv2.inRange(hsv, lower, upper)
        kernel = np.ones((5,5), np.uint8)
        mascara_color = cv2.morphologyEx(mascara_color, cv2.MORPH_OPEN, kernel)
        mascara_color = cv2.morphologyEx(mascara_color, cv2.MORPH_CLOSE, kernel)
        mascara_color = cv2.GaussianBlur(mascara_color, (5,5), 0)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
        mascara_3ch = cv2.cvtColor(mascara_color, cv2.COLOR_GRAY2BGR) / 255.0
        resultado = (frame * mascara_3ch + frame_gray * (1 - mascara_3ch)).astype(np.uint8)
        return resultado
        
    def actualizar_frame(self):
        if not self.camara_activa: return
        ret, frame = self.cap.read()
        if ret:
            self.frame_actual = self.aplicar_efecto_cine(frame)
            rgb = cv2.cvtColor(self.frame_actual, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            pixmap = pixmap.scaled(self.label_video.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.label_video.setPixmap(pixmap)
            
    def guardar_instantanea(self):
        if self.camara_activa and hasattr(self, 'frame_actual'):
            timestamp = QDateTime.currentDateTime().toString("yyyyMMdd_hhmmss")
            cv2.imwrite(f"captura_cine_{timestamp}.png", self.frame_actual)
            
    def closeEvent(self, event):
        if self.camara_activa:
            self.cap.release()
        event.accept()

def main():
    app = QApplication(sys.argv)
    ventana = SelectorColorMagico()
    ventana.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()