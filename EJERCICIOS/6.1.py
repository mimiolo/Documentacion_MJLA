import sys
import cv2
import mediapipe as mp
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QComboBox, QGroupBox)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap

class MallaFacialArtistica(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Malla Facial Artística - Capítulo 6")
        self.setGeometry(100, 100, 1000, 600)
        
        # Carga normal de MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1, 
            refine_landmarks=True, 
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.estilo_dibujo = "Tesselation (Malla)"
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
        
        grupo_estilo = QGroupBox("Estilo de Malla")
        layout_estilo = QVBoxLayout()
        self.combo_estilo = QComboBox()
        self.combo_estilo.addItems(["Tesselation (Malla)", "Contornos", "Puntos (Iris)"])
        self.combo_estilo.currentTextChanged.connect(self.cambiar_estilo)
        layout_estilo.addWidget(self.combo_estilo)
        grupo_estilo.setLayout(layout_estilo)
        layout_control.addWidget(grupo_estilo)
        
        self.info_label = QLabel("Rostros detectados: 0")
        layout_control.addWidget(self.info_label)
        
        layout_control.addStretch()
        layout.addWidget(panel_control, 1)

    def cambiar_estilo(self, texto):
        self.estilo_dibujo = texto

    def actualizar_frame(self):
        ret, frame = self.cap.read()
        if not ret: return
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        resultados = self.face_mesh.process(rgb)
        
        if resultados.multi_face_landmarks:
            self.info_label.setText(f"Rostros detectados: {len(resultados.multi_face_landmarks)}")
            for face_landmarks in resultados.multi_face_landmarks:
                if self.estilo_dibujo == "Tesselation (Malla)":
                    self.mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style())
                elif self.estilo_dibujo == "Contornos":
                    self.mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style())
                elif self.estilo_dibujo == "Puntos (Iris)":
                    self.mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_iris_connections_style())
        else:
            self.info_label.setText("Rostros detectados: 0")

        rgb_disp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_disp.shape
        qt_image = QImage(rgb_disp.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.label_video.setPixmap(QPixmap.fromImage(qt_image).scaled(self.label_video.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def closeEvent(self, event):
        self.cap.release()
        self.face_mesh.close()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ventana = MallaFacialArtistica()
    ventana.show()
    sys.exit(app.exec())