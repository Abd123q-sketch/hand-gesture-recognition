import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque
import mediapipe as mp

# Charger le modèle
model = load_model('hand_gesture_cnn_lstm.h5')

# Paramètres
seq_length = 21
img_size = 64
gesture_names = ['first', 'like', 'okay', 'open_hand', 'peace', 'thumbs_up']

# Initialisation
frames_queue = deque(maxlen=seq_length)
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    hand_detected = False

    if results.multi_hand_landmarks:
        hand_detected = True
        hand = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        # --- EXTRACTION DE LA MAIN (bounding box) ---
        h, w, _ = frame.shape
        x_min = min([lm.x for lm in hand.landmark]) * w
        x_max = max([lm.x for lm in hand.landmark]) * w
        y_min = min([lm.y for lm in hand.landmark]) * h
        y_max = max([lm.y for lm in hand.landmark]) * h

        # marges
        x_min = max(0, int(x_min - 20))
        y_min = max(0, int(y_min - 20))
        x_max = min(w, int(x_max + 20))
        y_max = min(h, int(y_max + 20))

        # Extraire la main
        hand_img = frame[y_min:y_max, x_min:x_max]

    else:
        hand_detected = False
        hand_img = None

    if hand_detected and hand_img is not None and hand_img.size > 0:
        # --- PRÉTRAITEMENT IDENTIQUE À L'ENTRAÎNEMENT ---
        hand_img = cv2.resize(hand_img, (img_size, img_size))
        hand_img = hand_img.astype("float32") / 255.0

        frames_queue.append(hand_img)

        # Si on a une séquence complète
        if len(frames_queue) == seq_length:
            seq = np.array(frames_queue)
            seq = np.expand_dims(seq, axis=0)

            pred = model.predict(seq, verbose=0)
            class_id = np.argmax(pred)
            class_name = gesture_names[class_id]
            confidence = pred[0][class_id]

            cv2.putText(frame, f"{class_name} ({confidence*100:.1f}%)",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)
    else:
        frames_queue.clear()
        cv2.putText(frame, "No hand detected",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)

    cv2.imshow("Hand Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
