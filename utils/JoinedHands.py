
import cv2
import mediapipe as mp
from math import sqrt

mp_pose = mp.solutions.pose
pose_detect = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.8,
                           min_tracking_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils


def find_euclidean_dist(x1, x2, y1, y2):
    x_diff = abs(x1 - x2)
    y_diff = abs(y1 - y2)
    euclidean_dist = sqrt(x_diff ** 2 + y_diff ** 2)
    return euclidean_dist


def joined_hands(img, results, draw=False):
    hand_status = None
    height, width, channel = img.shape

    output = img.copy()

    left_wrist_landmark = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * width,
                           results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * height)
    right_wrist_landmark = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * width,
                            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * height)

    euclidean_distance = int(find_euclidean_dist(left_wrist_landmark[0], left_wrist_landmark[1],
                                                 right_wrist_landmark[0], right_wrist_landmark[1]))
    if euclidean_distance < 130:
        hand_status = 'joined'
    else:
        hand_status = 'not joined'

    if draw:
        cv2.putText(output, hand_status, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
        cv2.putText(output, f'Distance: {euclidean_distance}', (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)

    return output, hand_status
