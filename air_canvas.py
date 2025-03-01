import cv2
import numpy as np
import mediapipe as mp

# Initialize Mediapipe Hand Tracking
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Initialize Canvas
canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
cap = cv2.VideoCapture(0)
prev_x, prev_y = 0, 0
color = (255, 0, 0)  # Default color: Blue
brush_size = 5
is_drawing = False

def draw_on_canvas(x, y):
    global prev_x, prev_y, canvas, is_drawing
    if prev_x == 0 and prev_y == 0:
        prev_x, prev_y = x, y
    if is_drawing:
        cv2.line(canvas, (prev_x, prev_y), (x, y), color, brush_size)
    prev_x, prev_y = x, y

def clear_canvas():
    global canvas
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

def change_color(x, y):
    global color
    if y < 50:  # Top bar for color selection
        if x < 200:
            color = (0, 0, 255)  # Red
        elif x < 400:
            color = (0, 255, 0)  # Green
        elif x < 600:
            color = (255, 0, 0)  # Blue
        elif x < 800:
            color = (0, 0, 0)  # Eraser

def draw_color_palette(frame):
    cv2.rectangle(frame, (0, 0), (200, 50), (0, 0, 255), -1)  # Red
    cv2.rectangle(frame, (200, 0), (400, 50), (0, 255, 0), -1)  # Green
    cv2.rectangle(frame, (400, 0), (600, 50), (255, 0, 0), -1)  # Blue
    cv2.rectangle(frame, (600, 0), (800, 50), (0, 0, 0), -1)  # Eraser

def detect_pinch(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    distance = np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) - np.array([index_tip.x, index_tip.y]))
    return distance < 0.05  # Adjust threshold as needed

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    draw_color_palette(frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            index_finger_tip = hand_landmarks.landmark[8]
            x, y = int(index_finger_tip.x * frame.shape[1]), int(index_finger_tip.y * frame.shape[0])

            if y < 50:
                change_color(x, y)
            elif detect_pinch(hand_landmarks):
                clear_canvas()
            else:
                is_drawing = index_finger_tip.z < -0.02
                draw_on_canvas(x, y)

    frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
    cv2.imshow("Air Canvas", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
