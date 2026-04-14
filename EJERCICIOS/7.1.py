import sys
import cv2
import mediapipe as mp
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QGroupBox,
                             QColorDialog, QSpinBox)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap

class PinturaDedos(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pintura con Dedos - Capítulo 7")
        self.setGeometry(100, 100, 1100, 700)
        
        # Configurar MediaPipe Hands de forma limpia
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1, 
            min_detection_confidence=0.7, 
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.lienzo = None
        self.ultima_posicion = None
        self.color_actual = (0, 255, 0) # Verde por defecto (BGR)
        self.grosor_actual = 5
        self.modo_borrador = False
        
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
        
        grupo_color = QGroupBox("Herramientas")
        layout_color = QVBoxLayout()
        
        btn_color = QPushButton("🎨 Seleccionar Color")
        btn_color.clicked.connect(self.seleccionar_color)
        layout_color.addWidget(btn_color)
        
        self.btn_borrador = QPushButton("🧽 Activar Borrador")
        self.btn_borrador.setCheckable(True)
        self.btn_borrador.toggled.connect(self.cambiar_borrador)
        layout_color.addWidget(self.btn_borrador)
        
        grupo_color.setLayout(layout_color)
        layout_control.addWidget(grupo_color)
        
        grupo_grosor = QGroupBox("Grosor del Pincel")
        layout_grosor = QVBoxLayout()
        self.spin_grosor = QSpinBox()
        self.spin_grosor.setRange(1, 30)
        self.spin_grosor.setValue(5)
        self.spin_grosor.valueChanged.connect(self.cambiar_grosor)
        layout_grosor.addWidget(self.spin_grosor)
        grupo_grosor.setLayout(layout_grosor)
        layout_control.addWidget(grupo_grosor)
        
        grupo_acciones = QGroupBox("Acciones")
        layout_acciones = QVBoxLayout()
        
        btn_limpiar = QPushButton("🗑️ Limpiar Lienzo")
        btn_limpiar.clicked.connect(self.limpiar_lienzo)
        layout_acciones.addWidget(btn_limpiar)
        
        btn_guardar = QPushButton("💾 Guardar Dibujo")
        btn_guardar.clicked.connect(self.guardar_dibujo)
        layout_acciones.addWidget(btn_guardar)
        
        grupo_acciones.setLayout(layout_acciones)
        layout_control.addWidget(grupo_acciones)
        
        layout_control.addStretch()
        layout.addWidget(panel_control, 1)

    def seleccionar_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            # PyQt devuelve RGB, OpenCV usa BGR
            self.color_actual = (color.blue(), color.green(), color.red())
            self.btn_borrador.setChecked(False)

    def cambiar_borrador(self, estado):
        self.modo_borrador = estado

    def cambiar_grosor(self, valor):
        self.grosor_actual = valor

    def limpiar_lienzo(self):
        if self.lienzo is not None:
            self.lienzo = np.zeros_like(self.lienzo)

    def guardar_dibujo(self):
        if self.lienzo is not None:
            cv2.imwrite("mi_dibujo_ar.png", self.lienzo)

    def detectar_gesto(self, hand_landmarks):
        # Coordenadas de las puntas y nudillos
        punta_indice = hand_landmarks.landmark[8].y
        nudillo_indice = hand_landmarks.landmark[6].y
        
        punta_medio = hand_landmarks.landmark[12].y
        nudillo_medio = hand_landmarks.landmark[10].y
        
        punta_anular = hand_landmarks.landmark[16].y
        nudillo_anular = hand_landmarks.landmark[14].y
        
        punta_menique = hand_landmarks.landmark[20].y
        nudillo_menique = hand_landmarks.landmark[18].y
        
        # Verificar qué dedos están extendidos (punta más arriba que nudillo)
        indice_ext = punta_indice < nudillo_indice
        medio_ext = punta_medio < nudillo_medio
        anular_ext = punta_anular < nudillo_anular
        menique_ext = punta_menique < nudillo_menique
        
        if indice_ext and not medio_ext and not anular_ext and not menique_ext: 
            return "dibujar"
        elif indice_ext and medio_ext and not anular_ext and not menique_ext: 
            return "borrador"
        elif indice_ext and medio_ext and anular_ext and menique_ext: 
            return "limpiar"
        return "nada"

    def actualizar_frame(self):
        ret, frame = self.cap.read()
        if not ret: return
        
        frame = cv2.flip(frame, 1) # Efecto espejo
        h, w = frame.shape[:2]
        
        # Inicializar lienzo negro con el mismo tamaño que la cámara
        if self.lienzo is None or self.lienzo.shape != frame.shape:
            self.lienzo = np.zeros_like(frame)
            
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultados = self.hands.process(rgb)
        
        if resultados.multi_hand_landmarks:
            for hand_landmarks in resultados.multi_hand_landmarks:
                # Obtener coordenadas del dedo índice
                indice_x = int(hand_landmarks.landmark[8].x * w)
                indice_y = int(hand_landmarks.landmark[8].y * h)
                posicion = (indice_x, indice_y)
                
                gesto = self.detectar_gesto(hand_landmarks)
                
                if gesto == "dibujar":
                    color = (0, 0, 0) if self.modo_borrador else self.color_actual
                    grosor = self.grosor_actual * 3 if self.modo_borrador else self.grosor_actual
                    
                    if self.ultima_posicion is not None:
                        cv2.line(self.lienzo, self.ultima_posicion, posicion, color, grosor)
                    cv2.circle(self.lienzo, posicion, grosor//2, color, -1)
                    self.ultima_posicion = posicion
                    
                    # Dibujar un círculo en el dedo como feedback visual
                    cv2.circle(frame, posicion, 10, self.color_actual, 2)
                    
                elif gesto == "borrador":
                    if self.ultima_posicion is not None:
                        cv2.line(self.lienzo, self.ultima_posicion, posicion, (0, 0, 0), self.grosor_actual * 4)
                    cv2.circle(self.lienzo, posicion, self.grosor_actual * 2, (0, 0, 0), -1)
                    self.ultima_posicion = posicion
                    
                    # Círculo blanco grande para el borrador
                    cv2.circle(frame, posicion, 20, (255, 255, 255), 2)
                    
                elif gesto == "limpiar":
                    self.limpiar_lienzo()
                    self.ultima_posicion = None
                else:
                    self.ultima_posicion = None
                    
                # Dibujar esqueleto de la mano (opcional)
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        else:
            self.ultima_posicion = None
            
        # Combinar cámara con el lienzo
        # La cámara tiene peso 1, el lienzo peso 1, donde hay color en el lienzo se sobrepone
        frame_final = cv2.addWeighted(frame, 1, self.lienzo, 1, 0)
        
        # Mostrar en la interfaz
        rgb_disp = cv2.cvtColor(frame_final, cv2.COLOR_BGR2RGB)
        qt_image = QImage(rgb_disp.data, w, h, ch * w, QImage.Format.Format_RGB888) if (ch:=rgb_disp.shape[2]) else None
        self.label_video.setPixmap(QPixmap.fromImage(qt_image).scaled(self.label_video.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def closeEvent(self, event):
        self.cap.release()
        self.hands.close()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ventana = PinturaDedos()
    ventana.show()
    sys.exit(app.exec())