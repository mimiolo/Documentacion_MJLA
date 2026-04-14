import sys
import cv2
import mediapipe as mp
import numpy as np
import time
import math
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QGroupBox, 
                             QListWidget, QListWidgetItem, QSlider, QMessageBox)
from PyQt6.QtCore import Qt, QTimer, QDateTime
from PyQt6.QtGui import QImage, QPixmap

class FiltrosAnimados:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
        # Índices clave de MediaPipe
        self.indices = {"ojo_izq": 33, "ojo_der": 362, "frente_izq": 10, "frente_der": 338, "nariz": 1}
        self.filtros = self.crear_graficos_filtros()
        self.tiempo_inicio = time.time()
        
    def crear_graficos_filtros(self):
        """Genera imágenes RGBA para los filtros sin necesitar archivos externos"""
        filtros = {}
        # Gafas (RGBA)
        gafas = np.zeros((200, 400, 4), dtype=np.uint8)
        cv2.rectangle(gafas, (50, 80), (170, 120), (0, 0, 0, 200), -1) # Lente izq
        cv2.rectangle(gafas, (230, 80), (350, 120), (0, 0, 0, 200), -1) # Lente der
        cv2.rectangle(gafas, (170, 90), (230, 110), (0, 0, 0, 255), -1) # Puente
        filtros["gafas"] = gafas
        
        # Sombrero (RGBA)
        sombrero = np.zeros((300, 400, 4), dtype=np.uint8)
        cv2.rectangle(sombrero, (150, 50), (250, 150), (20, 20, 20, 255), -1) # Copa
        cv2.ellipse(sombrero, (200, 150), (120, 30), 0, 0, 360, (20, 20, 20, 255), -1) # Ala
        cv2.rectangle(sombrero, (150, 120), (250, 140), (0, 0, 255, 255), -1) # Cinta roja
        filtros["sombrero"] = sombrero
        return filtros

    def detectar(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)
        puntos = {}
        if res.multi_face_landmarks:
            landmarks = res.multi_face_landmarks[0].landmark
            h, w = frame.shape[:2]
            for nombre, idx in self.indices.items():
                puntos[nombre] = (int(landmarks[idx].x * w), int(landmarks[idx].y * h))
            return puntos, frame
        return None, frame

    def superponer_imagen(self, fondo, overlay, pos):
        """Superpone una imagen con canal Alpha (transparencia) sobre el fondo"""
        x, y = pos
        h, w = overlay.shape[:2]
        
        # Evitar salir de los límites de la pantalla
        if y < 0 or x < 0 or y + h > fondo.shape[0] or x + w > fondo.shape[1]: 
            return fondo
            
        if overlay.shape[2] == 4:
            alpha = overlay[:, :, 3] / 255.0
            for c in range(3):
                fondo[y:y+h, x:x+w, c] = fondo[y:y+h, x:x+w, c] * (1 - alpha) + overlay[:, :, c] * alpha
        return fondo

    def aplicar_filtro(self, frame, puntos, filtro, tamano_porcentaje, brillo):
        if not puntos: return frame
        
        if filtro == "gafas":
            ojo_izq, ojo_der = puntos["ojo_izq"], puntos["ojo_der"]
            cx, cy = (ojo_izq[0] + ojo_der[0]) // 2, (ojo_izq[1] + ojo_der[1]) // 2
            ancho = int(abs(ojo_der[0] - ojo_izq[0]) * 2.5 * (tamano_porcentaje / 100))
            alto = int(ancho * 0.3)
            if ancho > 0 and alto > 0:
                img = cv2.resize(self.filtros["gafas"], (ancho, alto))
                frame = self.superponer_imagen(frame, img, (cx - ancho//2, cy - alto//2))
            
        elif filtro == "sombrero":
            f_izq, f_der = puntos["frente_izq"], puntos["frente_der"]
            ancho = int(abs(f_der[0] - f_izq[0]) * 3.0 * (tamano_porcentaje / 100))
            alto = int(ancho * 0.75)
            if ancho > 0 and alto > 0:
                img = cv2.resize(self.filtros["sombrero"], (ancho, alto))
                frame = self.superponer_imagen(frame, img, ((f_izq[0]+f_der[0])//2 - ancho//2, f_izq[1] - alto + 20))
            
        elif filtro == "gafas_animadas":
            ojo_izq, ojo_der = puntos["ojo_izq"], puntos["ojo_der"]
            cx, cy = (ojo_izq[0] + ojo_der[0]) // 2, (ojo_izq[1] + ojo_der[1]) // 2
            ancho = int(abs(ojo_der[0] - ojo_izq[0]) * 2.5 * (tamano_porcentaje / 100))
            alto = int(ancho * 0.3)
            if ancho > 0 and alto > 0:
                img = np.zeros((alto, ancho, 4), dtype=np.uint8)
                t = time.time() - self.tiempo_inicio
                # Color que cambia con el tiempo
                color = (int(128+127*math.sin(t)), int(128+127*math.sin(t+2)), int(128+127*math.sin(t+4)), 200)
                cv2.rectangle(img, (ancho//4, alto//4), (ancho//2, 3*alto//4), color, -1)
                cv2.rectangle(img, (ancho//2, alto//4), (3*ancho//4, 3*alto//4), color, -1)
                frame = self.superponer_imagen(frame, img, (cx - ancho//2, cy - alto//2))

        if brillo != 0:
            frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=brillo)
        return frame

class FiltrosSnapchat(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎭 Filtros SnapAR - Capítulo 13")
        self.setGeometry(100, 100, 1200, 700)
        
        self.animador = FiltrosAnimados()
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.actualizar_frame)
        self.timer.start(30)
        
        self.filtro_actual = "gafas"
        self.tamano_filtro = 100
        self.brillo_filtro = 0
        self.setup_ui()

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        
        panel_video = QWidget()
        layout_video = QVBoxLayout(panel_video)
        self.label_video = QLabel()
        self.label_video.setMinimumSize(800, 600)
        self.label_video.setStyleSheet("border: 3px solid #444; background-color: #111;")
        self.label_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout_video.addWidget(self.label_video)
        
        btn_capturar = QPushButton("📸 Capturar Foto")
        btn_capturar.clicked.connect(self.capturar_foto)
        layout_video.addWidget(btn_capturar)
        layout.addWidget(panel_video, 3)
        
        panel_control = QWidget()
        panel_control.setMaximumWidth(300)
        layout_control = QVBoxLayout(panel_control)
        
        self.lista_filtros = QListWidget()
        filtros = [("😎 Gafas oscuras", "gafas"), ("🎩 Sombrero elegante", "sombrero"), ("🌈 Gafas RGB", "gafas_animadas")]
        for nombre, id_filtro in filtros:
            item = QListWidgetItem(nombre)
            item.setData(Qt.ItemDataRole.UserRole, id_filtro)
            self.lista_filtros.addItem(item)
        self.lista_filtros.currentItemChanged.connect(self.cambiar_filtro)
        layout_control.addWidget(self.lista_filtros)
        
        grupo_ajustes = QGroupBox("⚙️ Ajustes del Filtro")
        layout_ajustes = QVBoxLayout()
        
        layout_ajustes.addWidget(QLabel("Tamaño:"))
        slider_tamano = QSlider(Qt.Orientation.Horizontal)
        slider_tamano.setRange(50, 200)
        slider_tamano.setValue(100)
        slider_tamano.valueChanged.connect(lambda v: setattr(self, 'tamano_filtro', v))
        layout_ajustes.addWidget(slider_tamano)
        
        layout_ajustes.addWidget(QLabel("Brillo:"))
        slider_brillo = QSlider(Qt.Orientation.Horizontal)
        slider_brillo.setRange(-50, 50)
        slider_brillo.setValue(0)
        slider_brillo.valueChanged.connect(lambda v: setattr(self, 'brillo_filtro', v))
        layout_ajustes.addWidget(slider_brillo)
        
        grupo_ajustes.setLayout(layout_ajustes)
        layout_control.addWidget(grupo_ajustes)
        
        layout_control.addStretch()
        layout.addWidget(panel_control, 1)
        self.lista_filtros.setCurrentRow(0)

    def cambiar_filtro(self, current, previous):
        if current: 
            self.filtro_actual = current.data(Qt.ItemDataRole.UserRole)

    def capturar_foto(self):
        if hasattr(self, 'ultimo_frame'):
            nombre = f"selfie_ar_{QDateTime.currentDateTime().toString('yyyyMMdd_hhmmss')}.png"
            cv2.imwrite(nombre, self.ultimo_frame)
            QMessageBox.information(self, "¡Foto Capturada!", f"Se ha guardado tu selfie en:\n{nombre}")

    def actualizar_frame(self):
        ret, frame = self.cap.read()
        if not ret: return
        
        # Voltear como un espejo
        frame = cv2.flip(frame, 1)
        
        puntos, frame = self.animador.detectar(frame)
        if puntos:
            frame = self.animador.aplicar_filtro(frame, puntos, self.filtro_actual, self.tamano_filtro, self.brillo_filtro)
        self.ultimo_frame = frame.copy()
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qt_img = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[2]*rgb.shape[1], QImage.Format.Format_RGB888)
        self.label_video.setPixmap(QPixmap.fromImage(qt_img).scaled(self.label_video.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ventana = FiltrosSnapchat()
    ventana.show()
    sys.exit(app.exec())