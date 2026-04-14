import sys
import cv2
import cv2.aruco as aruco
import numpy as np
import json
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QGroupBox, 
                             QFileDialog, QMessageBox, QComboBox)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap

class CamaraCalibrada(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("📷 Cámara Calibrada - Capítulo 12")
        self.setGeometry(100, 100, 1400, 800)
        
        self.matriz_camara = None
        self.dist_coefs = None
        self.usar_calibracion = False
        
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.actualizar_frame)
        self.timer.start(30)
        
        self.modo_comparacion = "lado_a_lado"
        self.setup_ui()

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        
        panel_video = QWidget()
        layout_video = QVBoxLayout(panel_video)
        self.label_video = QLabel()
        self.label_video.setMinimumSize(900, 600)
        self.label_video.setStyleSheet("border: 2px solid #333; background-color: #111;")
        self.label_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout_video.addWidget(self.label_video)
        layout.addWidget(panel_video, 3)
        
        panel_control = QWidget()
        panel_control.setMaximumWidth(350)
        layout_control = QVBoxLayout(panel_control)
        
        grupo_carga = QGroupBox("📂 Calibración")
        layout_carga = QVBoxLayout()
        btn_cargar = QPushButton("Cargar parámetros...")
        btn_cargar.clicked.connect(self.cargar_parametros)
        layout_carga.addWidget(btn_cargar)
        self.label_estado = QLabel("❌ No calibrada")
        self.label_estado.setStyleSheet("color: red; font-weight: bold;")
        layout_carga.addWidget(self.label_estado)
        grupo_carga.setLayout(layout_carga)
        layout_control.addWidget(grupo_carga)
        
        grupo_modo = QGroupBox("👁️ Visualización")
        layout_modo = QVBoxLayout()
        self.combo_modo = QComboBox()
        self.combo_modo.addItems(["lado_a_lado", "deslizante", "comparacion_directa"])
        self.combo_modo.currentTextChanged.connect(lambda v: setattr(self, 'modo_comparacion', v))
        layout_modo.addWidget(self.combo_modo)
        self.cb_calibracion = QPushButton("🔧 Activar calibración")
        self.cb_calibracion.setCheckable(True)
        self.cb_calibracion.toggled.connect(self.toggle_calibracion)
        layout_modo.addWidget(self.cb_calibracion)
        grupo_modo.setLayout(layout_modo)
        layout_control.addWidget(grupo_modo)
        
        grupo_info = QGroupBox("ℹ️ Parámetros")
        layout_info = QVBoxLayout()
        self.info_text = QLabel("Matriz cámara:\n--\n\nDistorsión:\n--\n\nResolución: --")
        layout_info.addWidget(self.info_text)
        grupo_info.setLayout(layout_info)
        layout_control.addWidget(grupo_info)
        
        layout_control.addStretch()
        layout.addWidget(panel_control, 1)

    def cargar_parametros(self):
        archivo, _ = QFileDialog.getOpenFileName(self, "Cargar parámetros", "", "NPZ Files (*.npz);;JSON Files (*.json)")
        if archivo:
            try:
                if archivo.endswith('.npz'):
                    datos = np.load(archivo)
                    self.matriz_camara = datos['matriz_camara']
                    self.dist_coefs = datos['dist_coefs']
                else:
                    with open(archivo, 'r') as f:
                        datos = json.load(f)
                    self.matriz_camara = np.array(datos['matriz_camara'])
                    self.dist_coefs = np.array(datos['dist_coefs'])
                self.label_estado.setText("✅ Calibrada")
                self.label_estado.setStyleSheet("color: green; font-weight: bold;")
                self.actualizar_info()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"No se pudieron cargar los parámetros: {e}")

    def toggle_calibracion(self, activado):
        self.usar_calibracion = activado and self.matriz_camara is not None
        if activado and self.matriz_camara is None:
            self.cb_calibracion.setChecked(False)
            QMessageBox.warning(self, "Atención", "Primero debes cargar los parámetros de calibración")

    def actualizar_info(self):
        if self.matriz_camara is not None:
            fx, fy = self.matriz_camara[0, 0], self.matriz_camara[1, 1]
            cx, cy = self.matriz_camara[0, 2], self.matriz_camara[1, 2]
            k1, k2, p1, p2, k3 = self.dist_coefs.ravel()[:5]
            info = f"Matriz cámara:\nfx: {fx:.1f}, fy: {fy:.1f}\ncx: {cx:.1f}, cy: {cy:.1f}\n\nDistorsión:\nk1: {k1:.3f}, k2: {k2:.3f}\np1: {p1:.3f}, p2: {p2:.3f}\nk3: {k3:.3f}"
            self.info_text.setText(info)

    def corregir_frame(self, frame):
        if not self.usar_calibracion or self.matriz_camara is None: return frame
        h, w = frame.shape[:2]
        nueva_matriz, roi = cv2.getOptimalNewCameraMatrix(self.matriz_camara, self.dist_coefs, (w, h), 1, (w, h))
        frame_corregido = cv2.undistort(frame, self.matriz_camara, self.dist_coefs, None, nueva_matriz)
        x, y, w, h = roi
        return frame_corregido[y:y+h, x:x+w]

    def visualizar_comparacion(self, frame_original, frame_corregido):
        if frame_corregido is None: return frame_original
        h1, w1 = frame_original.shape[:2]
        h2, w2 = frame_corregido.shape[:2]
        
        if self.modo_comparacion == "lado_a_lado":
            frame_corregido = cv2.resize(frame_corregido, (int(w2 * (h1 / h2)), h1)) if h1 != h2 else frame_corregido
            combinado = np.hstack([frame_original, frame_corregido])
            cv2.putText(combinado, "ORIGINAL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(combinado, "CORREGIDA", (w1 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return combinado
        elif self.modo_comparacion == "deslizante":
            pos = w1 // 2
            frame_corregido_redim = cv2.resize(frame_corregido, (w1, h1))
            resultado = frame_original.copy()
            resultado[:, pos:] = frame_corregido_redim[:, pos:]
            cv2.line(resultado, (pos, 0), (pos, h1), (255, 255, 255), 3)
            return resultado
        else: # comparacion_directa
            self._mostrar_original = not getattr(self, '_mostrar_original', False)
            return frame_original if self._mostrar_original else frame_corregido

    def actualizar_frame(self):
        ret, frame = self.cap.read()
        if not ret: return
        frame_corregido = self.corregir_frame(frame) if self.usar_calibracion else None
        
        if self.usar_calibracion and frame_corregido is not None:
            frame_mostrar = self.visualizar_comparacion(frame, frame_corregido)
        else:
            frame_mostrar = frame.copy()
            cv2.putText(frame_mostrar, "SIN CALIBRAR", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        rgb = cv2.cvtColor(frame_mostrar, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_image = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.label_video.setPixmap(QPixmap.fromImage(qt_image).scaled(self.label_video.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ventana = CamaraCalibrada()
    ventana.show()
    sys.exit(app.exec())