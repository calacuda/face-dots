"""
face_cam.py

a filter to replace my face with feature tracking.


By: calacuda | MIT Licence | epoch: March 29, 2021
"""


import mediapipe as mp
import cv2
import pyvirtualcam


mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(1)
fmt = pyvirtualcam.PixelFormat.BGR

# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic, pyvirtualcam.Camera(1280, 720, 20, fmt=fmt) as camera:
    while cap.isOpened():
        try:
            ret, frame = cap.read()
            bg_img = cv2.imread("green.jpg")
            # Recolor Feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Make Detections
            results = holistic.process(image)
            # print(results.face_landmarks)
            # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks
            # Recolor image back to BGR for rendering
            #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # 1. Draw face landmarks
            mp_drawing.draw_landmarks(bg_img, results.face_landmarks, mp_holistic.FACE_CONNECTIONS, 
                                      mp_drawing.DrawingSpec(color=(240,32,160), thickness=2, circle_radius=1),
                                      mp_drawing.DrawingSpec(color=(240,32,160), thickness=2, circle_radius=1)
                                      )
            # 2. Right hand
            mp_drawing.draw_landmarks(bg_img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                      mp_drawing.DrawingSpec(color=(25,179,255), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(25,179,255), thickness=2, circle_radius=2)
                                      )
            # 3. Left Hand
            mp_drawing.draw_landmarks(bg_img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                      mp_drawing.DrawingSpec(color=(25,179,255), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(25,179,255), thickness=2, circle_radius=2)
                                      )
            # 4. Pose Detections
            mp_drawing.draw_landmarks(bg_img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                      mp_drawing.DrawingSpec(color=(25,25,255), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(25,25,255), thickness=2, circle_radius=2)
                                      )
            # cv2.imshow('Face Mesh', bg_img)
            # cv2.imshow('Face Mesh', image)
            camera.send(bg_img)
            camera.sleep_until_next_frame()
        except KeyboardInterrupt:
            break
        # if cv2.waitKey(10) & 0xFF == ord('q'):
        #     break


cap.release()
cv2.destroyAllWindows()
