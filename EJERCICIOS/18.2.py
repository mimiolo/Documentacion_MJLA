import sys
import cv2
import math
import numpy as np
import mediapipe as mp
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, 
                             QVBoxLayout, QLabel, QPushButton, QProgressBar, QGroupBox)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap, QFont

class AIGymTracker(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🏋️ AI Gym Tracker - Capítulo 18")
        self.setGeometry(100, 100, 1100, 700)
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        self.cap = cv2.VideoCapture(0)
        
        # Variables de gimnasio
        self.contador_reps = 0
        self.estado = "abajo" # 'arriba' o 'abajo'
        self.porcentaje = 0
        
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
        self.label_video.setStyleSheet("background-color: #111; border: 3px solid #4CAF50;")
        layout.addWidget(self.label_video, 3)
        
        panel_derecho = QWidget()
        layout_panel = QVBoxLayout(panel_derecho)
        
        titulo = QLabel("🦾 Curl de Bíceps")
        titulo.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        layout_panel.addWidget(titulo, alignment=Qt.AlignmentFlag.AlignCenter)
        
        grupo_datos = QGroupBox("Estadísticas")
        l_datos = QVBoxLayout()
        
        self.lbl_contador = QLabel("0")
        self.lbl_contador.setFont(QFont("Arial", 48, QFont.Weight.Bold))
        self.lbl_contador.setStyleSheet("color: #4CAF50;")
        self.lbl_contador.setAlignment(Qt.AlignmentFlag.AlignCenter)
        l_datos.addWidget(QLabel("Repeticiones:"))
        l_datos.addWidget(self.lbl_contador)
        
        l_datos.addWidget(QLabel("Progreso del movimiento:"))
        self.barra_progreso = QProgressBar()
        self.barra_progreso.setRange(0, 100)
        self.barra_progreso.setValue(0)
        self.barra_progreso.setStyleSheet("QProgressBar::chunk { background-color: #2196F3; }")
        l_datos.addWidget(self.barra_progreso)
        
        grupo_datos.setLayout(l_datos)
        layout_panel.addWidget(grupo_datos)
        
        btn_reiniciar = QPushButton("🔄 Reiniciar Contador")
        btn_reiniciar.setMinimumHeight(40)
        btn_reiniciar.clicked.connect(self.reiniciar)
        layout_panel.addWidget(btn_reiniciar)
        
        layout_panel.addStretch()
        layout.addWidget(panel_derecho, 1)

    def reiniciar(self):
        self.contador_reps = 0
        self.lbl_contador.setText(str(self.contador_reps))

    def calcular_angulo(self, p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        angulo = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        angulo = abs(angulo)
        if angulo > 180.0:
            angulo = 360.0 - angulo
        return int(angulo)

    def actualizar_frame(self):
        ret, frame = self.cap.read()
        if not ret: return
        
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)
        
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            
            # Detectar brazo derecho (Índices 12, 14, 16)
            hombro = (int(lm[12].x * w), int(lm[12].y * h))
            codo = (int(lm[14].x * w), int(lm[14].y * h))
            muneca = (int(lm[16].x * w), int(lm[16].y * h))
            
            # Dibujar esqueleto del brazo
            cv2.line(frame, hombro, codo, (255, 255, 255), 4)
            cv2.line(frame, codo, muneca, (255, 255, 255), 4)
            for pt in [hombro, codo, muneca]:
                cv2.circle(frame, pt, 8, (255, 100, 0), -1)
                
            angulo = self.calcular_angulo(hombro, codo, muneca)
            
            # Mapear el ángulo (aprox. 160 abajo, 30 arriba) a porcentaje (0-100)
            self.porcentaje = np.interp(angulo, (30, 160), (100, 0))
            self.barra_progreso.setValue(int(self.porcentaje))
            
            # Lógica de conteo
            if self.porcentaje == 100 and self.estado == "abajo":
                self.estado = "arriba"
            if self.porcentaje == 0 and self.estado == "arriba":
                self.estado = "abajo"
                self.contador_reps += 1
                self.lbl_contador.setText(str(self.contador_reps))
                
            # Feedback visual de ángulo en pantalla
            color_ang = (0, 255, 0) if self.porcentaje > 80 else (255, 255, 255)
            cv2.putText(frame, f"{int(angulo)}", (codo[0] - 50, codo[1] + 50), 
                        cv2.FONT_HERSHEY_DUPLEX, 1.5, color_ang, 2)
            
        rgb_res = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qt_img = QImage(rgb_res.data, rgb_res.shape[1], rgb_res.shape[0], rgb_res.shape[2]*rgb_res.shape[1], QImage.Format.Format_RGB888)
        self.label_video.setPixmap(QPixmap.fromImage(qt_img).scaled(self.label_video.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    v = AIGymTracker()
    v.show()
    sys.exit(app.exec())