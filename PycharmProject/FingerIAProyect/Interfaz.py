import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import os
from DetectorManos import DetectorManos
import numpy as np


class Interfaz:
    def __init__(self, modelo=None):
        self.detector = DetectorManos()
        self.modelo = modelo

    def cargar_dataset(self, ruta_dataset):
        X = []
        y = []
        for etiqueta in os.listdir(ruta_dataset):
            ruta_clase = os.path.join(ruta_dataset, etiqueta)
            for nombre_imagen in os.listdir(ruta_clase):
                ruta_imagen = os.path.join(ruta_clase, nombre_imagen)
                imagen = cv2.imread(ruta_imagen)
                caracteristicas = self.detector.extraer_caracteristicas(imagen)
                if caracteristicas is not None:
                    X.append(caracteristicas)
                    y.append(int(etiqueta))
        return np.array(X), np.array(y)

    def entrenar_modelo(self, X_train, y_train):
        self.modelo = RandomForestClassifier(n_estimators=100, random_state=42)
        self.modelo.fit(X_train, y_train)

    def predecir(self, imagen):
        caracteristicas = self.detector.extraer_caracteristicas(imagen)
        if caracteristicas is not None:
            return self.modelo.predict([caracteristicas])[0]
        return None

    def evaluar_modelo(self, X_test, y_test):
        y_pred = self.modelo.predict(X_test)
        precision = precision_score(y_test, y_pred)
        return precision

    def iniciar_interfaz(self):
        captura = cv2.VideoCapture(0)
        while captura.isOpened():
            success, frame = captura.read()
            if not success:
                break

            resultado = self.detector.detectar_manos(frame)
            self.detector.dibujar_manos(frame, resultado)

            if resultado.multi_hand_landmarks:
                dedos_levantados = self.predecir(frame)
                if dedos_levantados is not None:
                    cv2.putText(frame, f'Dedos levantados: {dedos_levantados}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Reconocimiento de Gestos', frame)

            if cv2.waitKey(5) & 0xFF == 27:
                break

        captura.release()
        cv2.destroyAllWindows()
