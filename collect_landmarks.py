import cv2
import mediapipe as mp
import numpy as np
import os
import csv
import argparse
import tkinter as tk
from tkinter import simpledialog

# Caminho para o modelo do MediaPipe
MODEL_PATH = "models/gesture_recognizer.task"

def main():
    parser = argparse.ArgumentParser(description='Coleta de hand landmarks para dataset.')
    parser.add_argument('--label', type=str, default='', help='Label do gesto que será coletado')
    parser.add_argument('--output', type=str, default='dataset/gestures.csv', help='Arquivo CSV de saída')
    args = parser.parse_args()

    # Se a label não foi fornecida via linha de comando, pedir via GUI
    if not args.label:
        root = tk.Tk()
        root.withdraw()
        initial_label = simpledialog.askstring("Gesto Inicial", "Digite o nome do primeiro gesto que deseja gravar:", parent=root)
        root.destroy()
        
        if initial_label and initial_label.strip():
            args.label = initial_label.strip()
        else:
            print("Nenhuma label informada. Encerrando o script.")
            return

    # Verifica se o modelo existe na pasta
    if not os.path.exists(MODEL_PATH):
        print(f"Erro: O arquivo {MODEL_PATH} não foi encontrado na pasta atual.")
        return

    # Certifica que a pasta de saída existe
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Inicializa o modelo do MediaPipe Tasks
    BaseOptions = mp.tasks.BaseOptions
    GestureRecognizer = mp.tasks.vision.GestureRecognizer
    GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Configurações do Reconhecedor de Gestos
    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2, # Permitir a detecção de até duas mãos simultaneamente
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Prepara o arquivo CSV
    file_exists = os.path.isfile(args.output)
    
    # Inicia a captura da Webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Utilidades de desenho
    mp_hands = mp.tasks.vision.HandLandmarksConnections
    mp_drawing = mp.tasks.vision.drawing_utils
    mp_drawing_styles = mp.tasks.vision.drawing_styles

    print(f"Coletando dados para a label: {args.label}")
    print("Comandos:")
    print("  's' - Salva UMA captura dos landmarks atuais")
    print("  'r' - Inicia/Para gravação CONTÍNUA")
    print("  'n' - Digitar NOVO nome de gesto (Abrirá uma caixinha)")
    print("  'q' - Sair")

    recording = False

    with GestureRecognizer.create_from_options(options) as recognizer:
        with open(args.output, mode='a', newline='') as f:
            csv_writer = csv.writer(f)
            
            # Se o arquivo for novo, NÃO VAMOS escrever o cabeçalho original deste script 
            # porque o formato antigo do projeto não usa cabeçalho textual.
            # O formato que o script original espera no treinamento usa apenas: [label, x0, y0, z0, ...]
            # Mas vamos adaptar para salvar com [label] ao invés da string inteira como prefixo para facilitar parse se necessário.

            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
                
                recognition_result = recognizer.recognize_for_video(mp_image, timestamp_ms)

                if recognition_result.hand_landmarks:
                    for i, current_landmarks in enumerate(recognition_result.hand_landmarks):
                        hand_label = recognition_result.handedness[i][0].category_name
                        handedness_val = 0 if hand_label == 'Left' else 1
                        
                        # Desenha os landmarks
                        mp_drawing.draw_landmarks(
                            frame,
                            current_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())

                        # Se estiver gravando continuamente e houver landmarks, salva
                        if recording:
                            row = [args.label, handedness_val]
                            for lm in current_landmarks:
                                row.extend([lm.x, lm.y, lm.z])
                            csv_writer.writerow(row)
                        
                        cv2.putText(frame, f"Mao detectada: {hand_label}", (20, 60 + (i * 30)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Nenhuma mao detectada", (20, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                cv2.imshow('Coleta de Dados - Landmark Recorder', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    if recognition_result.hand_landmarks:
                        for i, current_landmarks in enumerate(recognition_result.hand_landmarks):
                            hand_label = recognition_result.handedness[i][0].category_name
                            handedness_val = 0 if hand_label == 'Left' else 1
                            
                            row = [args.label, handedness_val]
                            for lm in current_landmarks:
                                row.extend([lm.x, lm.y, lm.z])
                            csv_writer.writerow(row)
                        
                        f.flush()
                        print(f"Frame único salvo para label '{args.label}' ({len(recognition_result.hand_landmarks)} mao(s))")
                        cv2.rectangle(frame, (0,0), (640,480), (0,255,0), 10)
                        cv2.imshow('Coleta de Dados - Landmark Recorder', frame)
                        cv2.waitKey(50)
                elif key == ord('r'):
                    recording = not recording
                    print(f"Gravação contínua: {'INICIADA' if recording else 'PARADA'}")
                elif key == ord('n'):
                    was_recording = recording
                    recording = False
                    
                    root = tk.Tk()
                    root.withdraw()
                    new_label = simpledialog.askstring("Novo Gesto", "Digite o nome do novo gesto:", parent=root)
                    root.destroy()
                    
                    if new_label and new_label.strip():
                        args.label = new_label.strip()
                        print(f"\n---> Agora coletando dados para a NOVA label: '{args.label}' <---")
                    else:
                        print("Cancelado. Mantendo a label atual.")
                        
                    if was_recording:
                        print("Atenção: A gravação contínua foi pausada. Aperte 'r' para retomar.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
