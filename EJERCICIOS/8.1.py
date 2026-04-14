import sys
import cv2
import mediapipe as mp
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QGroupBox,
                             QComboBox, QProgressBar)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap

class ContadorEjercicios(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Contador de Ejercicios - Capítulo 8")
        self.setGeometry(100, 100, 1100, 700)
        
        # Configuración limpia de MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.ejercicio_actual = "sentadilla"
        self.contador = 0
        self.etapa = "arriba" # Estado inicial
        self.umbral_abajo = 90
        self.umbral_arriba = 160
        
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
        
        grupo_ejercicio = QGroupBox("🏋️‍♂️ Ejercicio")
        layout_ejercicio = QVBoxLayout()
        self.combo_ejercicio = QComboBox()
        self.combo_ejercicio.addItems(["sentadilla", "flexion", "abdominal"])
        self.combo_ejercicio.currentTextChanged.connect(self.cambiar_ejercicio)
        layout_ejercicio.addWidget(self.combo_ejercicio)
        grupo_ejercicio.setLayout(layout_ejercicio)
        layout_control.addWidget(grupo_ejercicio)
        
        grupo_contador = QGroupBox("📈 Progreso")
        layout_contador = QVBoxLayout()
        self.label_contador = QLabel("0")
        self.label_contador.setStyleSheet("font-size: 72px; font-weight: bold; color: #4CAF50;")
        self.label_contador.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout_contador.addWidget(self.label_contador)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout_contador.addWidget(self.progress_bar)
        grupo_contador.setLayout(layout_contador)
        layout_control.addWidget(grupo_contador)
        
        grupo_estado = QGroupBox("📊 Estado Actual")
        layout_estado = QVBoxLayout()
        self.label_etapa = QLabel("Fase: Arriba")
        self.label_etapa.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout_estado.addWidget(self.label_etapa)
        
        self.label_angulo = QLabel("Ángulo: --")
        self.label_angulo.setStyleSheet("font-size: 16px;")
        layout_estado.addWidget(self.label_angulo)
        grupo_estado.setLayout(layout_estado)
        layout_control.addWidget(grupo_estado)
        
        btn_reset = QPushButton("🔄 Reiniciar Contador")
        btn_reset.clicked.connect(self.reiniciar_contador)
        layout_control.addWidget(btn_reset)
        
        layout_control.addStretch()
        layout.addWidget(panel_control, 1)

    def cambiar_ejercicio(self, ejercicio):
        self.ejercicio_actual = ejercicio
        self.reiniciar_contador()

    def reiniciar_contador(self):
        self.contador = 0
        self.etapa = "arriba"
        self.label_contador.setText("0")
        self.progress_bar.setValue(0)

    def calcular_angulo(self, a, b, c, landmarks, w, h):
        punto_a = np.array([landmarks[a].x * w, landmarks[a].y * h])
        punto_b = np.array([landmarks[b].x * w, landmarks[b].y * h])
        punto_c = np.array([landmarks[c].x * w, landmarks[c].y * h])
        
        ba = punto_a - punto_b
        bc = punto_c - punto_b
        
        cos_angulo = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angulo = np.degrees(np.arccos(np.clip(cos_angulo, -1.0, 1.0)))
        
        return angulo, tuple(punto_a.astype(int)), tuple(punto_b.astype(int)), tuple(punto_c.astype(int))

    def contar_ejercicio(self, angulo):
        if angulo < self.umbral_abajo and self.etapa == "arriba":
            self.etapa = "abajo"
        elif angulo > self.umbral_arriba and self.etapa == "abajo":
            self.etapa = "arriba"
            self.contador += 1
            self.label_contador.setText(str(self.contador))
            # Simula una meta de 100 para la barra de progreso
            self.progress_bar.setValue(min(self.contador * 5, 100))

    def actualizar_frame(self):
        ret, frame = self.cap.read()
        if not ret: return
        
        # Para ejercicios es mejor NO hacer el flip espejo para que izquierda sea izquierda
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultados = self.pose.process(rgb)
        h, w = frame.shape[:2]
        
        if resultados.pose_landmarks:
            self.mp_drawing.draw_landmarks(frame, resultados.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            
            # Puntos clave según el ejercicio
            try:
                if self.ejercicio_actual == "sentadilla":
                    # Cadera (23), Rodilla (25), Tobillo (27)
                    angulo, a, b, c = self.calcular_angulo(23, 25, 27, resultados.pose_landmarks.landmark, w, h)
                elif self.ejercicio_actual == "flexion":
                    # Hombro (11), Codo (13), Muñeca (15)
                    angulo, a, b, c = self.calcular_angulo(11, 13, 15, resultados.pose_landmarks.landmark, w, h)
                else: # abdominal
                    # Hombro (11), Cadera (23), Rodilla (25)
                    angulo, a, b, c = self.calcular_angulo(11, 23, 25, resultados.pose_landmarks.landmark, w, h)
                    
                # Dibujar las líneas del ángulo
                cv2.circle(frame, a, 8, (255, 0, 0), -1)
                cv2.circle(frame, b, 8, (0, 255, 0), -1)
                cv2.circle(frame, c, 8, (0, 0, 255), -1)
                cv2.line(frame, a, b, (255, 255, 0), 3)
                cv2.line(frame, b, c, (255, 255, 0), 3)
                
                cv2.putText(frame, f"Angulo: {int(angulo)}", (b[0] + 20, b[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                self.label_angulo.setText(f"Ángulo: {angulo:.1f}°")
                self.label_etapa.setText(f"Fase: {self.etapa.capitalize()}")
                self.contar_ejercicio(angulo)
                
            except Exception as e:
                pass # Ignorar si no se ven todos los puntos en cámara
            
        rgb_disp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qt_image = QImage(rgb_disp.data, w, h, ch * w, QImage.Format.Format_RGB888) if (ch:=rgb_disp.shape[2]) else None
        self.label_video.setPixmap(QPixmap.fromImage(qt_image).scaled(self.label_video.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def closeEvent(self, event):
        self.cap.release()
        self.pose.close()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ventana = ContadorEjercicios()
    ventana.show()
    sys.exit(app.exec())