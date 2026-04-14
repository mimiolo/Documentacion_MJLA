import sys
import cv2
import cv2.aruco as aruco
import numpy as np
import mediapipe as mp
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QGroupBox, 
                             QSlider, QComboBox)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap

class CuboARInteractivo(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎲 Cubo AR Interactivo - Capítulo 11")
        self.setGeometry(100, 100, 1400, 800)
        
        # Configurar detector ArUco
        self.diccionario = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        self.parametros = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.diccionario, self.parametros)
        
        # Parámetros de cámara
        self.matriz_camara = np.array([[1000, 0, 640],
                                       [0, 1000, 360],
                                       [0, 0, 1]], dtype=np.float32)
        self.dist_coefs = np.zeros((4, 1))
        
        # Tamaños
        self.tamanio_marcador = 0.05  # 5 cm
        self.lado_cubo = 0.03  # 3 cm
        
        # Vértices y caras del cubo
        self.cubo_3d = self.crear_cubo_3d(self.lado_cubo)
        self.caras_cubo = self.definir_caras_cubo()
        
        # Variables de interacción
        self.rotacion_x = 0
        self.rotacion_y = 0
        self.rotacion_auto = True
        self.color_actual = (0, 255, 0)  # Verde
        self.modo_color = "sólido"
        
        # Detector de manos para interacción
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        
        # Captura de video
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.actualizar_frame)
        self.timer.start(30)
        
        self.setup_ui()

    def crear_cubo_3d(self, lado):
        l = lado / 2
        return np.float32([
            [-l, -l, 0], [l, -l, 0], [l, l, 0], [-l, l, 0],  # Base
            [-l, -l, l], [l, -l, l], [l, l, l], [-l, l, l]   # Tapa
        ])

    def definir_caras_cubo(self):
        return [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [0, 3, 7, 4]]

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
        layout.addWidget(panel_video, 3)
        
        panel_control = QWidget()
        panel_control.setMaximumWidth(350)
        layout_control = QVBoxLayout(panel_control)
        
        grupo_rotacion = QGroupBox("🔄 Rotación")
        layout_rotacion = QVBoxLayout()
        btn_auto = QPushButton("Auto/Manual")
        btn_auto.setCheckable(True)
        btn_auto.setChecked(True)
        btn_auto.toggled.connect(lambda v: setattr(self, 'rotacion_auto', v))
        layout_rotacion.addWidget(btn_auto)
        
        layout_rotacion.addWidget(QLabel("Rotación X:"))
        slider_rx = QSlider(Qt.Orientation.Horizontal)
        slider_rx.setRange(-180, 180)
        slider_rx.valueChanged.connect(lambda v: setattr(self, 'rotacion_x', v))
        layout_rotacion.addWidget(slider_rx)
        
        layout_rotacion.addWidget(QLabel("Rotación Y:"))
        slider_ry = QSlider(Qt.Orientation.Horizontal)
        slider_ry.setRange(-180, 180)
        slider_ry.valueChanged.connect(lambda v: setattr(self, 'rotacion_y', v))
        layout_rotacion.addWidget(slider_ry)
        grupo_rotacion.setLayout(layout_rotacion)
        layout_control.addWidget(grupo_rotacion)
        
        grupo_color = QGroupBox("🎨 Color")
        layout_color = QVBoxLayout()
        self.combo_color = QComboBox()
        self.combo_color.addItems(["sólido", "arcoíris", "por cara", "distancia"])
        self.combo_color.currentTextChanged.connect(lambda v: setattr(self, 'modo_color', v))
        layout_color.addWidget(self.combo_color)
        grupo_color.setLayout(layout_color)
        layout_control.addWidget(grupo_color)
        
        grupo_info = QGroupBox("ℹ️ Info")
        layout_info = QVBoxLayout()
        self.info_label = QLabel("🖐️ Gestos:\n- Puño: pausa rotación\n- Índice extendido: modo arcoíris\n- Mano abierta: modo sólido")
        layout_info.addWidget(self.info_label)
        grupo_info.setLayout(layout_info)
        layout_control.addWidget(grupo_info)
        
        layout_control.addStretch()
        layout.addWidget(panel_control, 1)

    def detectar_gesto(self, hand_landmarks):
        if not hand_landmarks: return "none"
        
        punta_indice = hand_landmarks.landmark[8]
        punta_medio = hand_landmarks.landmark[12]
        punta_anular = hand_landmarks.landmark[16]
        punta_menique = hand_landmarks.landmark[20]
        nudillo_indice = hand_landmarks.landmark[6]
        
        if (punta_indice.y < nudillo_indice.y and punta_medio.y > nudillo_indice.y and
            punta_anular.y > nudillo_indice.y and punta_menique.y > nudillo_indice.y):
            return "indice"
        elif (punta_indice.y < nudillo_indice.y and punta_medio.y < nudillo_indice.y):
            return "paz"
        elif all(p.y < nudillo_indice.y for p in [punta_indice, punta_medio, punta_anular, punta_menique]):
            return "abierta"
        return "none"

    def dibujar_cubo_coloreado(self, img, puntos_2d):
        puntos = np.int32(puntos_2d).reshape(-1, 2)
        if self.modo_color == "sólido":
            overlay = img.copy()
            for cara in self.caras_cubo:
                pts_cara = np.array([puntos[i] for i in cara], np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(overlay, [pts_cara], self.color_actual)
            cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
            self.dibujar_aristas(img, puntos, (255, 255, 255), 2)
        elif self.modo_color == "arcoíris":
            colores = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
            for i, cara in enumerate(self.caras_cubo):
                pts_cara = np.array([puntos[j] for j in cara], np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(img, [pts_cara], colores[i % len(colores)])
            self.dibujar_aristas(img, puntos, (255, 255, 255), 1)
        elif self.modo_color == "por cara":
            for i, cara in enumerate(self.caras_cubo):
                color = (50 + i*40, 100 + i*20, 150 + i*30)
                pts_cara = np.array([puntos[j] for j in cara], np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(img, [pts_cara], color)
            self.dibujar_aristas(img, puntos, (0, 0, 0), 2)
        elif self.modo_color == "distancia":
            for i, cara in enumerate(self.caras_cubo):
                z_promedio = np.mean([self.cubo_3d_proyectado[j][2] for j in cara])
                intensidad = int(255 * (1 - z_promedio / 0.1))
                color = (intensidad, intensidad, max(0, 255 - intensidad))
                pts_cara = np.array([puntos[j] for j in cara], np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(img, [pts_cara], color)

    def dibujar_aristas(self, img, puntos, color, grosor):
        aristas = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
        for a, b in aristas:
            cv2.line(img, tuple(puntos[a]), tuple(puntos[b]), color, grosor)

    def actualizar_frame(self):
        ret, frame = self.cap.read()
        if not ret: return
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultados_hands = self.hands.process(rgb)
        gesto = "none"
        
        if resultados_hands.multi_hand_landmarks:
            for hand_landmarks in resultados_hands.multi_hand_landmarks:
                gesto = self.detectar_gesto(hand_landmarks)
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        
        if gesto == "paz":
            self.rotacion_auto = False
            self.rotacion_x += 2
        elif gesto == "indice":
            self.modo_color = "arcoíris"
        elif gesto == "abierta":
            self.modo_color = "sólido"
            self.color_actual = (0, 255, 0)
            
        esquinas, ids, _ = self.detector.detectMarkers(frame)
        if ids is not None:
            aruco.drawDetectedMarkers(frame, esquinas, ids)
            for i in range(len(ids)):
                obj_points = np.array([[-self.tamanio_marcador/2, self.tamanio_marcador/2, 0],
                                       [self.tamanio_marcador/2, self.tamanio_marcador/2, 0],
                                       [self.tamanio_marcador/2, -self.tamanio_marcador/2, 0],
                                       [-self.tamanio_marcador/2, -self.tamanio_marcador/2, 0]], dtype=np.float32)
                success, rvec, tvec = cv2.solvePnP(obj_points, esquinas[i][0], self.matriz_camara, self.dist_coefs)
                if success:
                    if self.rotacion_auto:
                        self.rotacion_y = (self.rotacion_y + 2) % 360
                    R_extra, _ = cv2.Rodrigues(np.array([self.rotacion_x * np.pi/180, self.rotacion_y * np.pi/180, 0]))
                    self.cubo_3d_proyectado = np.dot(self.cubo_3d, R_extra.T)
                    imgpts, _ = cv2.projectPoints(self.cubo_3d_proyectado, rvec, tvec, self.matriz_camara, self.dist_coefs)
                    self.dibujar_cubo_coloreado(frame, imgpts)
                    cv2.drawFrameAxes(frame, self.matriz_camara, self.dist_coefs, rvec, tvec, 0.03)
                    
        self.mostrar_imagen(frame)

    def mostrar_imagen(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_image = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.label_video.setPixmap(QPixmap.fromImage(qt_image).scaled(self.label_video.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def closeEvent(self, event):
        self.cap.release()
        self.hands.close()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ventana = CuboARInteractivo()
    ventana.show()
    sys.exit(app.exec())