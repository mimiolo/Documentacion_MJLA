import sys
import cv2
import json
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QGroupBox,
                             QListWidget, QListWidgetItem, QInputDialog, QMessageBox)
from PyQt6.QtCore import Qt, QTimer, QDateTime
from PyQt6.QtGui import QImage, QPixmap

class DetectorAsistencia(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Detector de Asistencia - Capítulo 5")
        self.setGeometry(100, 100, 1100, 700)
        
        ruta_cascada = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(ruta_cascada)
        
        self.personas_conocidas = self.cargar_personas()
        self.setup_ui()
        
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.actualizar_frame)
        self.timer.start(30)

    def cargar_personas(self):
        if os.path.exists('personas.json'):
            try:
                with open('personas.json', 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        return {}

    def guardar_personas(self):
        with open('personas.json', 'w') as f:
            json.dump(self.personas_conocidas, f, indent=2)

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        
        panel_video = QWidget()
        layout_video = QVBoxLayout(panel_video)
        self.label_video = QLabel()
        self.label_video.setMinimumSize(640, 480)
        self.label_video.setStyleSheet("border: 2px solid #333; background-color: #111;")
        self.label_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout_video.addWidget(self.label_video)
        
        self.info_label = QLabel("Esperando detecciones...")
        layout_video.addWidget(self.info_label)
        layout.addWidget(panel_video, 3)
        
        panel_control = QWidget()
        panel_control.setMaximumWidth(350)
        layout_control = QVBoxLayout(panel_control)
        
        grupo_asistencia = QGroupBox("Asistencia Actual")
        layout_asistencia = QVBoxLayout()
        self.lista_asistencia = QListWidget()
        layout_asistencia.addWidget(self.lista_asistencia)
        
        btn_registrar = QPushButton("Registrar nueva persona")
        btn_registrar.clicked.connect(self.registrar_persona)
        layout_asistencia.addWidget(btn_registrar)
        grupo_asistencia.setLayout(layout_asistencia)
        layout_control.addWidget(grupo_asistencia)
        
        grupo_bd = QGroupBox("Base de Datos")
        layout_bd = QVBoxLayout()
        self.lista_bd = QListWidget()
        layout_bd.addWidget(self.lista_bd)
        
        btn_eliminar = QPushButton("Eliminar seleccionado")
        btn_eliminar.clicked.connect(self.eliminar_persona)
        layout_bd.addWidget(btn_eliminar)
        grupo_bd.setLayout(layout_bd)
        layout_control.addWidget(grupo_bd)
        
        layout_control.addStretch()
        layout.addWidget(panel_control, 1)
        self.actualizar_lista_bd()

    def reconocer_persona(self, w, h):
        for nombre, datos in self.personas_conocidas.items():
            if abs(w - datos.get('ancho', 0)) < 40 and abs(h - datos.get('alto', 0)) < 40:
                return nombre
        return "Desconocido"

    def actualizar_frame(self):
        ret, frame = self.cap.read()
        if not ret: return
        
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rostros = self.face_cascade.detectMultiScale(gris, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        
        asistentes = []
        
        for (x, y, w, h) in rostros:
            nombre = self.reconocer_persona(w, h)
            asistentes.append(nombre)
            
            color = (0, 255, 0) if nombre != "Desconocido" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, nombre, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
        self.actualizar_lista_asistencia(asistentes, len(rostros))
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h_img, w_img, ch = rgb.shape
        qt_image = QImage(rgb.data, w_img, h_img, ch * w_img, QImage.Format.Format_RGB888)
        self.label_video.setPixmap(QPixmap.fromImage(qt_image).scaled(self.label_video.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def actualizar_lista_asistencia(self, asistentes, cantidad_rostros):
        self.lista_asistencia.clear()
        for nombre in set(asistentes):
            item = QListWidgetItem(nombre)
            item.setForeground(Qt.GlobalColor.red if nombre == "Desconocido" else Qt.GlobalColor.green)
            self.lista_asistencia.addItem(item)
        self.info_label.setText(f"Personas detectadas: {cantidad_rostros}")

    def actualizar_lista_bd(self):
        self.lista_bd.clear()
        for nombre in self.personas_conocidas.keys():
            self.lista_bd.addItem(nombre)

    def registrar_persona(self):
        ret, frame = self.cap.read()
        if not ret: return
        
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rostros = self.face_cascade.detectMultiScale(gris, 1.1, 5, minSize=(50, 50))
        
        if len(rostros) > 0:
            nombre, ok = QInputDialog.getText(self, "Registrar", "Nombre de la persona:")
            if ok and nombre:
                x, y, w, h = rostros[0]
                self.personas_conocidas[nombre] = {
                    'ancho': int(w), 
                    'alto': int(h), 
                    'fecha': QDateTime.currentDateTime().toString()
                }
                self.guardar_personas()
                self.actualizar_lista_bd()
                QMessageBox.information(self, "Éxito", "Persona registrada correctamente.")
        else:
            QMessageBox.warning(self, "Error", "No se detectó ningún rostro para registrar. Ubícate frente a la cámara.")

    def eliminar_persona(self):
        current = self.lista_bd.currentItem()
        if current:
            nombre = current.text()
            respuesta = QMessageBox.question(self, "Confirmar", f"¿Eliminar a {nombre}?")
            if respuesta == QMessageBox.StandardButton.Yes:
                del self.personas_conocidas[nombre]
                self.guardar_personas()
                self.actualizar_lista_bd()

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ventana = DetectorAsistencia()
    ventana.show()
    sys.exit(app.exec())