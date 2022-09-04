
import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose_detect = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.8,
                           min_tracking_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils


def pose_detection(img, draw=False):
    output = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = pose_detect.process(img)

    if result.pose_landmarks and draw:
        mp_drawing.draw_landmarks(image=output, landmark_list=result.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing.DrawingSpec(color=(200, 200, 200), thickness=2,
                                                                               circle_radius=5),
                                  connection_drawing_spec=mp_drawing.DrawingSpec(color=(80, 80, 80), thickness=2,
                                                                                 circle_radius=3))

    return output, result
