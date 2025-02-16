
import cv2
import mediapipe as mp
import time

# Подключаем камеру
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Ширина
cap.set(4, 480)  # Высота
cap.set(10, 100)  # Яркость

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

pTime = 0

while True:
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        ret, frame = cap.read()

        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Рисуем сетку на лице
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                  mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))

        # Рисуем линию на плечах
        if results.pose_landmarks:
            left_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
            cv2.line(image, (int(left_shoulder.x * image.shape[1]), int(left_shoulder.y * image.shape[0])),
                            (int(right_shoulder.x * image.shape[1]), int(right_shoulder.y * image.shape[0])), (0, 0, 255), 3)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(image, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        cv2.imshow('Python', image)

        if cv2.waitKey(1) == 27:  # Нажмите ESC для выхода
            break

cv2.destroyAllWindows()
cap.release()
