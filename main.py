import cv2
import mediapipe as mp
import numpy as np
import time
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
canvas = None

prev_time = 0

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
    prev_x, prev_y = 0, 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        if canvas is None:
            canvas = np.zeros((h, w, 3), np.uint8)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                x = int(hand.landmark[8].x * w)
                y = int(hand.landmark[8].y * h)

                cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)

                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x, y

                cv2.line(canvas, (prev_x, prev_y), (x, y), (255, 0, 0), 5)
                prev_x, prev_y = x, y

                mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
        else:
            prev_x, prev_y = 0, 0

        blended = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time
        cv2.putText(blended, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Air Drawing", blended)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
