
import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose_detect = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.8,
                           min_tracking_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils


def find_horizontal_posture(img, res, draw=False):
    status = None
    h, w, c = img.shape
    output = img.copy()

    left_x = int(res.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w)
    right_x = int(res.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w)

    if right_x <= w // 2 and left_x <= w // 2:
        status = 'l'
    elif right_x >= w // 2 and left_x >= w // 2:
        status = 'r'
    elif left_x >= w // 2 >= right_x:
        status = 'centre'

    if draw:
        cv2.putText(output, status, (50, h - 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 5)
        cv2.line(output, (w // 2, 0), (w // 2, h), (255, 255, 255), 3)

    return output, status
