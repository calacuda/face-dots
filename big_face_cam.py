#!/bin/env python
"""
face_cam.py

a filter to replace my face with feature tracking.


By: calacuda | MIT Licence | epoch: March 29, 2021
"""


import mediapipe as mp
import cv2
import pyvirtualcam
# from numba import jit


# Initiate holistic model

def main():
    # print("main")
    mp_drawing = mp.solutions.drawing_utils
    # mp_holistic = mp.solutions.holistic
    mp_face = mp.solutions.face_mesh
    cap = cv2.VideoCapture(2)
    
    fmt = pyvirtualcam.PixelFormat.RGB
    # bgf = "rainbow_fall.jpg"
    drawing_spec = mp_drawing.DrawingSpec(color=(255, 255, 255),  # color=(240, 32, 160),
                                          thickness=4,
                                          circle_radius=1)
    c = 0.3
    with mp_face.FaceMesh(min_detection_confidence=c, min_tracking_confidence=c) as model, \
         pyvirtualcam.Camera(1280, 720, 30, fmt=fmt, device="/dev/video1") as camera:
        print("model trained")
        while cap.isOpened():
            try:
                ret, frame = cap.read(cv2.COLOR_BGR2GRAY)
                
                face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
                # Detect faces
                faces = face_cascade.detectMultiScale(frame, 1.1, 4)
                for (x, y, w, h) in faces:
                    
                    cv2.rectangle(frame, (x, y), (x+w*2, y+h*2),
                                  (0, 0, 255), 2)
                    faces = frame[y:y + h * 2, x:x + w * 2]
                bg_img = cv2.imread("green.jpg")
                # Make Detections
                if type(faces) != tuple:
                    results = model.process(faces)
                # 1. Draw face landmarks
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        # print(face_landmarks)
                        mp_drawing.draw_landmarks(
                            image=bg_img,
                            landmark_list=face_landmarks,
                            connections=mp_face.FACE_CONNECTIONS,
                            landmark_drawing_spec=drawing_spec,
                            connection_drawing_spec=drawing_spec)
                    camera.send(bg_img)
                    camera.sleep_until_next_frame()
            except KeyboardInterrupt:
                break
            except UnboundLocalError:
                pass
            # if cv2.waitKey(10) & 0xFF == ord('q'):
            #     break
    print("releasing capture")
    cap.release()
    print("destroying windows")
    cv2.destroyAllWindows()
    print("done")


if __name__ == "__main__":
    main()
