import cv2
import mediapipe as mp
import numpy as np

# Inicializar MediaPipe Hands y Face Mesh
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Índices de landmarks para el mentón y la oreja
INDICE_MENTON = 152
INDICE_OREJA_DERECHA = 234

# Definir función para detectar si se está sosteniendo un teléfono cerca de la oreja
def detectar_hablando_por_telefono(hand_landmarks, face_landmarks, height, width):
    muñeca = [hand_landmarks[mp_hands.HandLandmark.WRIST].x * width, hand_landmarks[mp_hands.HandLandmark.WRIST].y * height]

    # Punto de referencia del mentón
    menton = [face_landmarks[INDICE_MENTON].x * width, face_landmarks[INDICE_MENTON].y * height]
    
    # Punto de referencia de la oreja
    oreja = [face_landmarks[INDICE_OREJA_DERECHA].x * width, face_landmarks[INDICE_OREJA_DERECHA].y * height]
    
    # Distancia de la muñeca a la oreja y al mentón
    dist_oreja = np.linalg.norm(np.array(muñeca) - np.array(oreja))
    dist_menton = np.linalg.norm(np.array(muñeca) - np.array(menton))

    # Verificar si la mano está cerca de la oreja (sosteniendo un teléfono)
    if dist_oreja < width * 0.15 and muñeca[1] < menton[1]:
        return True
    return False

# Captura de video desde la cámara
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Obtener dimensiones del frame
    height, width, _ = frame.shape

    # Convertir la imagen a RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Procesar manos y rostro
    result_hands = hands.process(image_rgb)
    result_face = face_mesh.process(image_rgb)

    if result_hands.multi_hand_landmarks and result_face.multi_face_landmarks:
        for hand_landmarks in result_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            hablando_por_telefono = detectar_hablando_por_telefono(
                hand_landmarks.landmark, 
                result_face.multi_face_landmarks[0].landmark, 
                height, 
                width
            )
            if hablando_por_telefono:
                cv2.putText(frame, "Hablando por telefono", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, "Hablando por telefono", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Detección de Hablando por Teléfono', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
