import sys
import cv2
import numpy as np
import random
import mediapipe as mp
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel, QStackedWidget
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap, QFont

class JuegoMotor:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.ancho, self.alto = 800, 600
        self.puntos, self.vidas = 0, 3
        self.objetos = []
        self.velocidad_base = 6
        self.juego_terminado = False

    def crear_objeto(self):
        # 80% probabilidad bueno, 20% malo
        tipo = 'bueno' if random.random() < 0.8 else 'malo'
        self.objetos.append({
            'x': random.randint(40, self.ancho - 40), 
            'y': -20, 
            'tipo': tipo,
            'radio': 25, 
            'puntos': 15 if tipo == 'bueno' else -20,
            'vel': self.velocidad_base * (1.5 if tipo == 'malo' else 1.0)
        })

    def procesar(self, frame):
        if self.juego_terminado: return frame
        
        frame = cv2.resize(cv2.flip(frame, 1), (self.ancho, self.alto))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)
        
        pos_mano = None
        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0]
            # Usamos el punto 9 (nudillo medio) como centro de la mano
            pos_mano = (int(lm.landmark[9].x * self.ancho), int(lm.landmark[9].y * self.alto))
            cv2.circle(frame, pos_mano, 35, (255, 255, 255), 3)
            
        # Generación aleatoria de objetos (máximo 4 en pantalla)
        if random.random() < 0.04 and len(self.objetos) < 4: 
            self.crear_objeto()
        
        activos = []
        for obj in self.objetos:
            obj['y'] += obj['vel']
            
            # Si se sale de la pantalla
            if obj['y'] > self.alto: 
                # Si dejas caer uno bueno, restas un punto como penalización leve
                if obj['tipo'] == 'bueno': self.puntos = max(0, self.puntos - 1)
                continue
                
            atrapado = False
            if pos_mano:
                dist = np.hypot(pos_mano[0] - obj['x'], pos_mano[1] - obj['y'])
                if dist < obj['radio'] + 35:
                    self.puntos = max(0, self.puntos + obj['puntos'])
                    if obj['tipo'] == 'malo': 
                        self.vidas -= 1
                    atrapado = True
                    
            if not atrapado: 
                activos.append(obj)
                
            color = (0, 255, 0) if obj['tipo'] == 'bueno' else (0, 0, 255)
            cv2.circle(frame, (int(obj['x']), int(obj['y'])), obj['radio'], color, -1)
            
        self.objetos = activos
        
        # UI en el frame
        cv2.putText(frame, f"Puntos: {self.puntos}", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,0), 2)
        cv2.putText(frame, f"Vidas: {'❤️' * self.vidas}", (self.ancho - 250, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,255), 2)
        
        if self.vidas <= 0:
            self.juego_terminado = True
            
        return frame

class ARCatcherApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎮 AR Catcher Completo - Capítulo 16")
        self.setGeometry(100, 100, 850, 650)
        
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        
        # --- PANTALLA MENÚ ---
        self.menu_widget = QWidget()
        self.menu_widget.setStyleSheet("background-color: #2b2b2b;")
        l_menu = QVBoxLayout(self.menu_widget)
        
        titulo = QLabel("🎮 AR CATCHER")
        titulo.setFont(QFont("Arial", 36, QFont.Weight.Bold))
        titulo.setStyleSheet("color: white;")
        titulo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        l_menu.addWidget(titulo)
        
        instrucciones = QLabel("Atrapa los 🟢 (Puntos)\nEvita los 🔴 (Daño)\n¡Usa tu mano frente a la cámara!")
        instrucciones.setStyleSheet("color: #aaaaaa; font-size: 18px;")
        instrucciones.setAlignment(Qt.AlignmentFlag.AlignCenter)
        l_menu.addWidget(instrucciones)
        
        btn_jugar = QPushButton("EMPEZAR JUEGO")
        btn_jugar.setFixedSize(200, 60)
        btn_jugar.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; border-radius: 10px;")
        btn_jugar.clicked.connect(self.iniciar_juego)
        l_menu.addWidget(btn_jugar, alignment=Qt.AlignmentFlag.AlignCenter)
        self.stack.addWidget(self.menu_widget)
        
        # --- PANTALLA JUEGO ---
        self.juego_widget = QWidget()
        self.juego_widget.setStyleSheet("background-color: black;")
        l_juego = QVBoxLayout(self.juego_widget)
        
        self.lbl_video = QLabel()
        self.lbl_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        l_juego.addWidget(self.lbl_video)
        
        btn_salir = QPushButton("Abandonar Partida")
        btn_salir.setStyleSheet("background-color: #f44336; color: white; padding: 10px;")
        btn_salir.clicked.connect(self.detener_juego)
        l_juego.addWidget(btn_salir)
        self.stack.addWidget(self.juego_widget)

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.actualizar_frame)

    def iniciar_juego(self):
        self.motor = JuegoMotor()
        self.cap = cv2.VideoCapture(0)
        self.timer.start(30)
        self.stack.setCurrentIndex(1) # Cambiar a la pantalla del juego

    def detener_juego(self):
        self.timer.stop()
        if self.cap: 
            self.cap.release()
        self.lbl_video.clear()
        self.stack.setCurrentIndex(0) # Volver al menú principal

    def actualizar_frame(self):
        ret, frame = self.cap.read()
        if not ret: return
        
        frame_procesado = self.motor.procesar(frame)
        
        # Dibujar Game Over si aplica
        if self.motor.juego_terminado:
            cv2.rectangle(frame_procesado, (0, 200), (800, 400), (0, 0, 0), -1)
            cv2.putText(frame_procesado, "GAME OVER", (150, 320), cv2.FONT_HERSHEY_DUPLEX, 3, (0,0,255), 5)
            self.timer.stop()
            # Esperar 3 segundos y volver al menú
            QTimer.singleShot(3000, self.detener_juego)
            
        rgb = cv2.cvtColor(frame_procesado, cv2.COLOR_BGR2RGB)
        qt_img = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[2]*rgb.shape[1], QImage.Format.Format_RGB888)
        self.lbl_video.setPixmap(QPixmap.fromImage(qt_img).scaled(self.lbl_video.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def closeEvent(self, event):
        self.detener_juego()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ventana = ARCatcherApp()
    ventana.show()
    sys.exit(app.exec())