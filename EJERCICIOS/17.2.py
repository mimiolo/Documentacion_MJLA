import sys
import cv2
import numpy as np
import mediapipe as mp
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, 
                             QVBoxLayout, QLabel, QPushButton, QSlider, QGroupBox, QComboBox)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap

class EstudioArteAR(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎨 Estudio de Arte AR - Capítulo 17")
        self.setGeometry(100, 100, 1100, 700)
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8)
        
        self.cap = cv2.VideoCapture(0)
        self.lienzo = None
        self.x_prev, self.y_prev = 0, 0
        
        self.color_actual = (255, 0, 0) # Azul en BGR
        self.grosor = 5
        self.dibujando = False
        
        self.setup_ui()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.actualizar_frame)
        self.timer.start(30)

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        
        self.label_video = QLabel()
        self.label_video.setMinimumSize(800, 600)
        self.label_video.setStyleSheet("background-color: black; border: 2px solid white;")
        layout.addWidget(self.label_video, 3)
        
        panel_derecho = QWidget()
        layout_panel = QVBoxLayout(panel_derecho)
        
        info = QLabel("🖐️ Controles:\n- 1 Dedo: Dibujar\n- 2 Dedos: Mover cursor\n- Mano cerrada: Pausa")
        info.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout_panel.addWidget(info)
        
        grupo_herramientas = QGroupBox("🖌️ Herramientas")
        l_herr = QVBoxLayout()
        
        l_herr.addWidget(QLabel("Color:"))
        self.combo_color = QComboBox()
        self.combo_color.addItems(["Azul", "Verde", "Rojo", "Amarillo", "Borrador"])
        self.combo_color.currentTextChanged.connect(self.cambiar_color)
        l_herr.addWidget(self.combo_color)
        
        l_herr.addWidget(QLabel("Grosor:"))
        self.slider_grosor = QSlider(Qt.Orientation.Horizontal)
        self.slider_grosor.setRange(2, 30)
        self.slider_grosor.setValue(5)
        self.slider_grosor.valueChanged.connect(lambda v: setattr(self, 'grosor', v))
        l_herr.addWidget(self.slider_grosor)
        
        btn_limpiar = QPushButton("🗑️ Limpiar Lienzo")
        btn_limpiar.clicked.connect(self.limpiar_lienzo)
        btn_limpiar.setStyleSheet("background-color: #ff4c4c; color: white;")
        l_herr.addWidget(btn_limpiar)
        
        grupo_herramientas.setLayout(l_herr)
        layout_panel.addWidget(grupo_herramientas)
        layout_panel.addStretch()
        
        layout.addWidget(panel_derecho, 1)

    def cambiar_color(self, texto):
        colores = {
            "Azul": (255, 0, 0), "Verde": (0, 255, 0),
            "Rojo": (0, 0, 255), "Amarillo": (0, 255, 255),
            "Borrador": (0, 0, 0)
        }
        self.color_actual = colores.get(texto, (255, 0, 0))

    def limpiar_lienzo(self):
        if self.lienzo is not None:
            self.lienzo = np.zeros_like(self.lienzo)

    def detectar_dedos_levantados(self, lm):
        """Retorna True si solo el índice está levantado"""
        puntas = [8, 12, 16, 20] # Índice, Medio, Anular, Meñique
        levantados = []
        
        for punta in puntas:
            # Si la punta está más alta (menor Y) que el nudillo inferior, está levantado
            if lm.landmark[punta].y < lm.landmark[punta - 2].y:
                levantados.append(1)
            else:
                levantados.append(0)
        return levantados

    def actualizar_frame(self):
        ret, frame = self.cap.read()
        if not ret: return
        
        frame = cv2.flip(frame, 1)
        if self.lienzo is None:
            self.lienzo = np.zeros_like(frame)
            
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)
        
        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0]
            dedos = self.detectar_dedos_levantados(lm)
            
            h, w = frame.shape[:2]
            x1, y1 = int(lm.landmark[8].x * w), int(lm.landmark[8].y * h) # Índice
            
            # Si solo índice está levantado -> Dibujar
            if dedos[0] == 1 and sum(dedos) == 1:
                cv2.circle(frame, (x1, y1), self.grosor, self.color_actual, cv2.FILLED)
                if self.x_prev == 0 and self.y_prev == 0:
                    self.x_prev, self.y_prev = x1, y1
                cv2.line(self.lienzo, (self.x_prev, self.y_prev), (x1, y1), self.color_actual, self.grosor)
                self.x_prev, self.y_prev = x1, y1
            
            # Si índice y medio están levantados -> Mover (no dibujar)
            elif dedos[0] == 1 and dedos[1] == 1 and sum(dedos) == 2:
                cv2.circle(frame, (x1, y1), self.grosor + 5, (200, 200, 200), 2) # Cursor modo mover
                self.x_prev, self.y_prev = 0, 0
                
            else:
                self.x_prev, self.y_prev = 0, 0
        else:
            self.x_prev, self.y_prev = 0, 0
            
        # Fusión limpia de lienzo y cámara
        img_gris = cv2.cvtColor(self.lienzo, cv2.COLOR_BGR2GRAY)
        _, mascara = cv2.threshold(img_gris, 10, 255, cv2.THRESH_BINARY_INV)
        fondo = cv2.bitwise_and(frame, frame, mask=mascara)
        resultado = cv2.bitwise_or(fondo, self.lienzo)
        
        rgb_res = cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB)
        qt_img = QImage(rgb_res.data, rgb_res.shape[1], rgb_res.shape[0], rgb_res.shape[2]*rgb_res.shape[1], QImage.Format.Format_RGB888)
        self.label_video.setPixmap(QPixmap.fromImage(qt_img).scaled(self.label_video.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    v = EstudioArteAR()
    v.show()
    sys.exit(app.exec())