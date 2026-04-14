import sys
import cv2
import numpy as np
import mediapipe as mp
import math
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QComboBox, QGroupBox
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap

class AlineadorObjetos:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
        self.catalogo = {
            'gafas': self.crear_gafas(),
            'bigote': self.crear_bigote()
        }

    def crear_gafas(self):
        gafas = np.zeros((150, 300, 4), dtype=np.uint8)
        cv2.rectangle(gafas, (30, 40), (120, 90), (50, 50, 50, 220), -1)
        cv2.rectangle(gafas, (180, 40), (270, 90), (50, 50, 50, 220), -1)
        cv2.rectangle(gafas, (120, 50), (180, 65), (20, 20, 20, 255), -1)
        return gafas

    def crear_bigote(self):
        bigote = np.zeros((80, 200, 4), dtype=np.uint8)
        cv2.ellipse(bigote, (60, 40), (40, 15), 20, 0, 360, (20, 10, 5, 255), -1)
        cv2.ellipse(bigote, (140, 40), (40, 15), -20, 0, 360, (20, 10, 5, 255), -1)
        return bigote

    def obtener_puntos(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        puntos = {}
        
        res = self.face_mesh.process(rgb)
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            puntos['ojo_izq'] = (int(lm[33].x * w), int(lm[33].y * h))
            puntos['ojo_der'] = (int(lm[362].x * w), int(lm[362].y * h))
            puntos['nariz'] = (int(lm[1].x * w), int(lm[1].y * h))
            puntos['boca_izq'] = (int(lm[61].x * w), int(lm[61].y * h))
            puntos['boca_der'] = (int(lm[291].x * w), int(lm[291].y * h))
        return puntos

    def superponer_imagen(self, fondo, overlay, x, y):
        h, w = overlay.shape[:2]
        if y < 0 or x < 0 or y + h > fondo.shape[0] or x + w > fondo.shape[1]: return fondo
        alpha = overlay[:, :, 3] / 255.0
        for c in range(3):
            fondo[y:y+h, x:x+w, c] = fondo[y:y+h, x:x+w, c] * (1 - alpha) + overlay[:, :, c] * alpha
        return fondo

    def aplicar(self, frame, obj):
        puntos = self.obtener_puntos(frame)
        if not puntos: return frame
        
        if obj == 'gafas' and 'ojo_izq' in puntos:
            oi, od = puntos['ojo_izq'], puntos['ojo_der']
            cx, cy = (oi[0] + od[0]) // 2, (oi[1] + od[1]) // 2
            dist = math.hypot(od[0] - oi[0], od[1] - oi[1])
            gafas = self.catalogo['gafas']
            esc = dist / (gafas.shape[1] * 0.35)
            if esc > 0:
                gafas_r = cv2.resize(gafas, (int(gafas.shape[1]*esc), int(gafas.shape[0]*esc)))
                frame = self.superponer_imagen(frame, gafas_r, cx - gafas_r.shape[1]//2, cy - gafas_r.shape[0]//2)
            
        elif obj == 'bigote' and 'nariz' in puntos:
            nariz, bi, bd = puntos['nariz'], puntos['boca_izq'], puntos['boca_der']
            cx, cy = nariz[0], (nariz[1] + (bi[1]+bd[1])//2) // 2
            dist = abs(bd[0] - bi[0])
            bigote = self.catalogo['bigote']
            esc = dist / (bigote.shape[1] * 0.3)
            if esc > 0:
                bigote_r = cv2.resize(bigote, (int(bigote.shape[1]*esc), int(bigote.shape[0]*esc)))
                frame = self.superponer_imagen(frame, bigote_r, cx - bigote_r.shape[1]//2, cy - bigote_r.shape[0]//2 - 10)
            
        return frame

class CatalogoVirtual(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("👗 Catálogo Virtual - Capítulo 15")
        self.setGeometry(100, 100, 1100, 700)
        
        self.alineador = AlineadorObjetos()
        self.objeto_activo = 'gafas'
        
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.actualizar_frame)
        self.timer.start(30)
        
        self.setup_ui()

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        
        self.label_video = QLabel()
        self.label_video.setMinimumSize(800, 600)
        self.label_video.setStyleSheet("background-color: #222;")
        self.label_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.label_video, 3)
        
        panel_derecho = QWidget()
        panel_derecho.setMaximumWidth(250)
        layout_panel = QVBoxLayout(panel_derecho)
        
        grupo = QGroupBox("🛍️ Accesorios")
        layout_grupo = QVBoxLayout()
        layout_grupo.addWidget(QLabel("Selecciona un artículo:"))
        
        self.combo = QComboBox()
        self.combo.addItems(["gafas", "bigote"])
        self.combo.currentTextChanged.connect(lambda v: setattr(self, 'objeto_activo', v))
        layout_grupo.addWidget(self.combo)
        
        grupo.setLayout(layout_grupo)
        layout_panel.addWidget(grupo)
        layout_panel.addStretch()
        
        layout.addWidget(panel_derecho, 1)

    def actualizar_frame(self):
        ret, frame = self.cap.read()
        if not ret: return
        
        frame = cv2.flip(frame, 1)
        frame = self.alineador.aplicar(frame, self.objeto_activo)
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qt_img = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[2]*rgb.shape[1], QImage.Format.Format_RGB888)
        self.label_video.setPixmap(QPixmap.fromImage(qt_img).scaled(self.label_video.size(), Qt.AspectRatioMode.KeepAspectRatio))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ventana = CatalogoVirtual()
    ventana.show()
    sys.exit(app.exec())