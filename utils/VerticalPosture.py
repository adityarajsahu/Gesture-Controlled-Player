
import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose_detect = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.8,
                           min_tracking_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils


def find_vertical_posture(img, res, shoulder_mid, draw=False):
    status = None
    h, w, c = img.shape
    output = img.copy()

    left_y = int(res.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h)
    right_y = int(res.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h)
    mid_point = abs(left_y + right_y) // 2

    lower_bound = shoulder_mid - 25
    upper_bound = shoulder_mid + 25

    if mid_point < lower_bound:
        status = 'j'
    elif mid_point > upper_bound:
        status = 'c'
    else:
        status = 's'

    if draw:
        cv2.putText(output, status, (50, h - 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 5)
        cv2.line(output, (0, mid_point), (w, mid_point), (255, 255, 0), 3)

    return output, status
