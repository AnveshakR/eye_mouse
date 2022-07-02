from turtle import left
import cv2
import numpy as np
import face_recognition

videopath = r"C:\Users\anves\Documents\Python Scripts\eye_video.mp4"

padding = 5

def detectAndDisplay(frame):
    frame = np.array(frame)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    face_locations = face_recognition.face_locations(frame,number_of_times_to_upsample=0,model='cnn')
    face_landmarks = face_recognition.face_landmarks(frame,face_locations,model="large")

    try:
        left_eye = np.array(face_landmarks[0]["left_eye"])
        # frame = cv2.polylines(frame, [left_eye], isClosed=True, color=(0,0,255), thickness=2)
        mask = np.ones(frame.shape, dtype=np.uint8)
        mask.fill(255)
        cv2.fillConvexPoly(mask, left_eye, 0)
        frame = cv2.bitwise_or(frame, mask)
        frame = frame[min(left_eye[:,1]):max(left_eye[:,1]), min(left_eye[:,0]):max(left_eye[:,0])]
        frame = cv2.resize(frame, (frame.shape[1]*2, frame.shape[0]*2))
        ret, thresh = cv2.threshold(frame, 30, 255, cv2.THRESH_BINARY_INV)
        thresh = cv2.resize(thresh, (160, 90))
        thresh = cv2.bitwise_not(thresh)
        cv2.imshow('output', thresh)
        return frame
    except:
        pass

cap = cv2.VideoCapture(3)
if not cap.isOpened:
    print('Error opening video capture')
    exit(0)
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame,(1280,720))
    if frame is None:
        pass
    
    detectAndDisplay(frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q' or 'Q'):
        break