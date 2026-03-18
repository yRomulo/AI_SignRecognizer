import cv2
import mediapipe as mp
import numpy as np
from .models import get_mediapipe_options, load_custom_models

class GestureProcessor:
    def __init__(self):
        # Carrega os modelos customizados
        self.clf, self.label_encoder = load_custom_models()
        self.options = get_mediapipe_options()
        
        # Inicializa o recognizer do MediaPipe
        # Usamos o context manager em __enter__ e __exit__ ou controlamos o ciclo
        self.recognizer = mp.tasks.vision.GestureRecognizer.create_from_options(self.options)
        
        # Utilitários de desenho do MediaPipe
        self.mp_hands = mp.tasks.vision.HandLandmarksConnections
        self.mp_drawing = mp.tasks.vision.drawing_utils
        self.mp_drawing_styles = mp.tasks.vision.drawing_styles

    def process_frame(self, frame, draw_landmarks=True, record_gesture=None):
        """
        Processa um frame (numpy array), detecta mãos, 
        faz predições e desenha landmarks no frame.
        Retorna o frame processado e a lista de labels.
        """
        # Inverte horizontalmente e converte para RGB (MediaPipe)
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # MediaPipe precisa de timestamp para o modo VIDEO
        timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        
        # Reconhece (extrai landmarks) usando MediaPipe
        recognition_result = self.recognizer.recognize_for_video(mp_image, timestamp_ms)

        labels = []
        if recognition_result.hand_landmarks:
            for i, hand_landmarks in enumerate(recognition_result.hand_landmarks):
                # 1. Desenha os landmarks (se habilitado)
                if draw_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )

                # Extrai features
                hand_label = recognition_result.handedness[i][0].category_name
                handedness_val = 0 if hand_label == 'Left' else 1
                
                landmarks_array = [handedness_val]
                for lm in hand_landmarks:
                    landmarks_array.extend([lm.x, lm.y, lm.z])
                
                features = np.array(landmarks_array).reshape(1, -1)

                if record_gesture:
                    # Save the landmarks to CSV instead of predicting
                    import csv
                    import os
                    os.makedirs('dataset', exist_ok=True)
                    with open('dataset/gestures.csv', 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([record_gesture] + landmarks_array)
                    
                    labels.append({
                        "hand": hand_label,
                        "gesture": f"Gravando: {record_gesture}",
                        "probability": 1.0
                    })
                else:
                    # Predição normal pelo modelo
                    prediction_idx = self.clf.predict(features)[0]
                    prediction_prob = np.max(self.clf.predict_proba(features))
                    gesture_name = self.label_encoder.inverse_transform([prediction_idx])[0]

                    labels.append({
                        "hand": hand_label,
                        "gesture": gesture_name,
                        "probability": float(prediction_prob)
                    })

        gesture_image = None
        if not record_gesture and len(labels) > 0:
            gesture_name = None
            if len(labels) == 1:
                gesture_name = labels[0]['gesture']
            elif len(labels) == 2 and labels[0]['gesture'] == labels[1]['gesture']:
                gesture_name = labels[0]['gesture']
                
            if gesture_name:
                import os
                image_filename = f"{gesture_name}.png"
                image_path = os.path.join('assets', 'images', 'gestures', image_filename)
                if os.path.exists(image_path):
                    gesture_image = image_filename

        return frame, labels, gesture_image

    def close(self):
        """Libera os recursos do recognizer."""
        if self.recognizer:
            self.recognizer.close()

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
