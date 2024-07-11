import cv2
import mediapipe as mp
import numpy as np

class DetectorManos:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

    def extraer_caracteristicas(self, imagen):
        resultado = self.hands.process(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
        if resultado.multi_hand_landmarks:
            hand_landmarks = resultado.multi_hand_landmarks[0]
            return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
        return None

    def dibujar_manos(self, imagen, resultado):
        if resultado.multi_hand_landmarks:
            for hand_landmarks in resultado.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(imagen, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

    def detectar_manos(self, imagen):
        return self.hands.process(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
