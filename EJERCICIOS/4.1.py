# Capitulo 4 - Mini Proyecto: Escaner de Documentos App
import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QGroupBox,
                             QSlider, QFileDialog, QMessageBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap, QAction
from datetime import datetime

class EscanerDocumentos(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Escaner de Documentos - Capitulo 4")
        self.setGeometry(100, 100, 1400, 800)
        self.imagen_original = None
        self.imagen_procesada = None
        self.puntos_documento = []
        self.modo_manual = False
        self.setup_ui()
        self.setup_menu()
        
    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        
        panel_imagenes = QWidget()
        layout_imagenes = QVBoxLayout(panel_imagenes)
        
        layout_imagenes.addWidget(QLabel("Original:"))
        self.label_original = QLabel()
        self.label_original.setMinimumSize(500, 400)
        self.label_original.setStyleSheet("border: 1px solid #333; background-color: #111;")
        self.label_original.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_original.mousePressEvent = self.seleccionar_punto
        layout_imagenes.addWidget(self.label_original)
        
        layout_imagenes.addWidget(QLabel("Enderezada:"))
        self.label_procesada = QLabel()
        self.label_procesada.setMinimumSize(500, 400)
        self.label_procesada.setStyleSheet("border: 1px solid #333; background-color: #111;")
        self.label_procesada.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout_imagenes.addWidget(self.label_procesada)
        
        layout.addWidget(panel_imagenes, 3)
        
        panel_control = QWidget()
        panel_control.setMaximumWidth(300)
        layout_control = QVBoxLayout(panel_control)
        
        grupo_carga = QGroupBox("Cargar imagen")
        layout_carga = QVBoxLayout()
        btn_cargar = QPushButton("Seleccionar imagen...")
        btn_cargar.clicked.connect(self.cargar_imagen)
        layout_carga.addWidget(btn_cargar)
        btn_webcam = QPushButton("Usar webcam")
        btn_webcam.clicked.connect(self.usar_webcam)
        layout_carga.addWidget(btn_webcam)
        grupo_carga.setLayout(layout_carga)
        layout_control.addWidget(grupo_carga)
        
        grupo_modo = QGroupBox("Modo de deteccion")
        layout_modo = QVBoxLayout()
        btn_auto = QPushButton("Automatico")
        btn_auto.clicked.connect(lambda: self.cambiar_modo(False))
        layout_modo.addWidget(btn_auto)
        btn_manual = QPushButton("Manual (seleccionar 4 puntos)")
        btn_manual.clicked.connect(lambda: self.cambiar_modo(True))
        layout_modo.addWidget(btn_manual)
        grupo_modo.setLayout(layout_modo)
        layout_control.addWidget(grupo_modo)
        
        grupo_ajustes = QGroupBox("Ajustes")
        layout_ajustes = QVBoxLayout()
        layout_ajustes.addWidget(QLabel("Umbral Canny 1:"))
        self.slider_canny1 = QSlider(Qt.Orientation.Horizontal)
        self.slider_canny1.setRange(0, 255)
        self.slider_canny1.setValue(50)
        self.slider_canny1.valueChanged.connect(self.actualizar_escaner)
        layout_ajustes.addWidget(self.slider_canny1)
        layout_ajustes.addWidget(QLabel("Umbral Canny 2:"))
        self.slider_canny2 = QSlider(Qt.Orientation.Horizontal)
        self.slider_canny2.setRange(0, 255)
        self.slider_canny2.setValue(150)
        self.slider_canny2.valueChanged.connect(self.actualizar_escaner)
        layout_ajustes.addWidget(self.slider_canny2)
        grupo_ajustes.setLayout(layout_ajustes)
        layout_control.addWidget(grupo_ajustes)
        
        grupo_acciones = QGroupBox("Acciones")
        layout_acciones = QVBoxLayout()
        btn_escanear = QPushButton("Escanear ahora")
        btn_escanear.clicked.connect(self.escanear_documento)
        layout_acciones.addWidget(btn_escanear)
        btn_guardar = QPushButton("Guardar resultado")
        btn_guardar.clicked.connect(self.guardar_resultado)
        layout_acciones.addWidget(btn_guardar)
        btn_mejorar = QPushButton("Mejorar imagen")
        btn_mejorar.clicked.connect(self.mejorar_imagen)
        layout_acciones.addWidget(btn_mejorar)
        grupo_acciones.setLayout(layout_acciones)
        layout_control.addWidget(grupo_acciones)
        
        layout_control.addStretch()
        layout.addWidget(panel_control, 1)
        
    def setup_menu(self):
        menubar = self.menuBar()
        archivo_menu = menubar.addMenu("&Archivo")
        abrir_action = QAction("&Abrir imagen", self)
        abrir_action.triggered.connect(self.cargar_imagen)
        archivo_menu.addAction(abrir_action)
        guardar_action = QAction("&Guardar resultado", self)
        guardar_action.triggered.connect(self.guardar_resultado)
        archivo_menu.addAction(guardar_action)
        salir_action = QAction("&Salir", self)
        salir_action.triggered.connect(self.close)
        archivo_menu.addAction(salir_action)
        
    def cargar_imagen(self):
        archivo, _ = QFileDialog.getOpenFileName(self, "Seleccionar imagen", "", "Imagenes (*.png *.jpg *.jpeg *.bmp *.tiff)")
        if archivo:
            self.imagen_original = cv2.imread(archivo)
            self.puntos_documento = []
            if self.imagen_original is not None:
                self.mostrar_imagen(self.imagen_original, self.label_original)
                if not self.modo_manual:
                    self.escanear_documento()
                    
    def usar_webcam(self):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if ret:
            self.imagen_original = frame
            self.puntos_documento = []
            self.mostrar_imagen(self.imagen_original, self.label_original)
            if not self.modo_manual:
                self.escanear_documento()
                
    def cambiar_modo(self, manual):
        self.modo_manual = manual
        self.puntos_documento = []
        if manual:
            if self.imagen_original is not None:
                self.mostrar_imagen(self.imagen_original, self.label_original)
            QMessageBox.information(self, "Modo manual", "Haz clic en 4 puntos de la imagen original en orden:\n1. Superior izquierdo\n2. Superior derecho\n3. Inferior derecho\n4. Inferior izquierdo")
        else:
            self.escanear_documento()
            
    def ordenar_puntos(self, puntos):
        puntos = puntos.reshape(4, 2)
        suma = puntos.sum(axis=1)
        diff = np.diff(puntos, axis=1)
        ordenados = np.zeros((4, 2), dtype=np.float32)
        ordenados[0] = puntos[np.argmin(suma)]
        ordenados[2] = puntos[np.argmax(suma)]
        ordenados[1] = puntos[np.argmin(diff)]
        ordenados[3] = puntos[np.argmax(diff)]
        return ordenados
        
    def detectar_documento_auto(self, imagen):
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        desenfoque = cv2.GaussianBlur(gris, (5, 5), 0)
        canny1 = self.slider_canny1.value()
        canny2 = self.slider_canny2.value()
        bordes = cv2.Canny(desenfoque, canny1, canny2)
        kernel = np.ones((5, 5), np.uint8)
        bordes = cv2.morphologyEx(bordes, cv2.MORPH_CLOSE, kernel)
        contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contornos: return None
        contorno_doc = max(contornos, key=cv2.contourArea)
        peri = cv2.arcLength(contorno_doc, True)
        aproximacion = cv2.approxPolyDP(contorno_doc, 0.02 * peri, True)
        if len(aproximacion) == 4: return aproximacion
        return None
        
    def escanear_documento(self):
        if self.imagen_original is None: return
        
        if self.modo_manual and len(self.puntos_documento) == 4:
            pts_origen = self.ordenar_puntos(np.array(self.puntos_documento, dtype=np.float32))
        elif not self.modo_manual:
            esquinas = self.detectar_documento_auto(self.imagen_original)
            if esquinas is None:
                QMessageBox.warning(self, "Error", "No se detecto automaticamente. Prueba en modo manual.")
                return
            pts_origen = self.ordenar_puntos(esquinas)
        else:
            return
            
        (tl, tr, br, bl) = pts_origen
        ancho1 = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        ancho2 = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        max_ancho = max(int(ancho1), int(ancho2))
        alto1 = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        alto2 = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_alto = max(int(alto1), int(alto2))
        pts_destino = np.array([[0, 0], [max_ancho - 1, 0], [max_ancho - 1, max_alto - 1], [0, max_alto - 1]], dtype=np.float32)
        H, _ = cv2.findHomography(pts_origen, pts_destino)
        self.imagen_procesada = cv2.warpPerspective(self.imagen_original, H, (max_ancho, max_alto))
        self.mostrar_imagen(self.imagen_procesada, self.label_procesada)
        
    def mejorar_imagen(self):
        if self.imagen_procesada is None: return
        gris = cv2.cvtColor(self.imagen_procesada, cv2.COLOR_BGR2GRAY)
        mejora = cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        self.imagen_procesada = cv2.cvtColor(mejora, cv2.COLOR_GRAY2BGR)
        self.mostrar_imagen(self.imagen_procesada, self.label_procesada)
        
    def actualizar_escaner(self):
        if self.imagen_original is not None and not self.modo_manual:
            self.escanear_documento()
            
    def guardar_resultado(self):
        if self.imagen_procesada is None: return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre = f"documento_escaner_{timestamp}.png"
        cv2.imwrite(nombre, self.imagen_procesada)
        QMessageBox.information(self, "Guardado", f"Imagen guardada como: {nombre}")
        
    def mostrar_imagen(self, imagen, label):
        rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image).scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        label.setPixmap(pixmap)
        
    def seleccionar_punto(self, event):
        if not self.modo_manual or self.imagen_original is None: return
        pixmap = self.label_original.pixmap()
        if pixmap is None: return
        
        ancho_lbl, alto_lbl = self.label_original.width(), self.label_original.height()
        ancho_pix, alto_pix = pixmap.width(), pixmap.height()
        margen_x = (ancho_lbl - ancho_pix) // 2
        margen_y = (alto_lbl - alto_pix) // 2
        
        x_click = event.pos().x() - margen_x
        y_click = event.pos().y() - margen_y
        
        if 0 <= x_click <= ancho_pix and 0 <= y_click <= alto_pix:
            h_img, w_img = self.imagen_original.shape[:2]
            x_real = int(x_click * (w_img / ancho_pix))
            y_real = int(y_click * (h_img / alto_pix))
            
            self.puntos_documento.append([x_real, y_real])
            img_feedback = self.imagen_original.copy()
            for pt in self.puntos_documento:
                cv2.circle(img_feedback, tuple(pt), 10, (0, 0, 255), -1)
            self.mostrar_imagen(img_feedback, self.label_original)
            
            if len(self.puntos_documento) == 4:
                self.escanear_documento()

def main():
    app = QApplication(sys.argv)
    ventana = EscanerDocumentos()
    ventana.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()