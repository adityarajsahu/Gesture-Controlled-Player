
import cv2
import mediapipe as mp
import pyautogui as pag
from utils.PoseDetection import pose_detection
from utils.JoinedHands import joined_hands
from utils.HorizontalPosture import find_horizontal_posture
from utils.VerticalPosture import find_vertical_posture
import tensorflow as tf

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 960)
cv2.namedWindow('Virtual Player', cv2.WINDOW_NORMAL)

started = False

curr_x_pos = 0
curr_y_pos = 0

y_shoulder_mid = None

count_frames_when_hand_joined = 0
count_frames_when_hand_joined_required = 15

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    height, width, channel = frame.shape
    frame, res = pose_detection(frame, draw=started)

    if res.pose_landmarks:
        if started:
            frame, horizontal_position = find_horizontal_posture(frame, res, draw=True)
            if (horizontal_position == 'l' and curr_x_pos != -1) or (horizontal_position == 'centre' and curr_x_pos == 1):
                pag.press("left")
                curr_x_pos -= 1
            elif (horizontal_position == 'r' and curr_x_pos != 1) or (horizontal_position == 'centre' and curr_x_pos == -1):
                pag.press("right")
                curr_x_pos += 1

        else:
            cv2.putText(frame, "JOIN HANDS TO START GAME", (50, height - 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 5)

        if joined_hands(frame, res)[1] == "joined":
            count_frames_when_hand_joined += 1
            if count_frames_when_hand_joined == count_frames_when_hand_joined_required:
                if not started:
                    started = True
                    left_y = int(res.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * height)
                    right_y = int(res.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * height)

                    y_shoulder_mid = abs(left_y + right_y) // 2
                    pag.click(x=1300, y=800, button='left')

                else:
                    pag.press('space')
                count_frames_when_hand_joined = 0

        else:
            count_frames_when_hand_joined = 0

        if y_shoulder_mid:
            frame, posture = find_vertical_posture(frame, res, y_shoulder_mid, draw=True)
            if posture == "j" and curr_y_pos == 0:
                pag.press('up')
                curr_y_pos += 1
            elif posture == "c" and curr_y_pos == 0:
                pag.press('down')
                curr_y_pos -= 1
            elif posture == "s" and curr_y_pos != 0:
                curr_y_pos = 0

    else:
        count_frames_when_hand_joined = 0

    cv2.imshow("Virtual Player", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
