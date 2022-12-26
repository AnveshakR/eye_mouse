import pyautogui
import cv2
import numpy as np
import face_recognition
from time import time

videopath = r"C:\Users\anves\Documents\Python Scripts\eye_video.mp4"
source = 1

display_width = pyautogui.size().width
display_height = pyautogui.size().height

def range_convert(oldval, oldmin, oldmax, newmin, newmax):
   newval = (((oldval - oldmin) * (newmax - newmin)) / (oldmax - oldmin)) + newmin
   return newval

def left_pupil_track(frame):

    frame = np.array(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    try:

        face_locations = face_recognition.face_locations(frame,number_of_times_to_upsample=0,model='cnn')
        face_landmarks = face_recognition.face_landmarks(frame,face_locations,model="large")

        left_eye = np.array(face_landmarks[0]["left_eye"])

        frame = cv2.polylines(frame, [left_eye], isClosed=True, color=(0,0,255), thickness=2)

        mask = np.ones(frame.shape, dtype=np.uint8)
        mask.fill(255)
        cv2.fillConvexPoly(mask, left_eye, 0)
        frame = cv2.bitwise_or(frame, mask)
        frame = frame[min(left_eye[:,1]):max(left_eye[:,1]), min(left_eye[:,0]):max(left_eye[:,0])]
        frame = cv2.resize(frame, (160, 90))
        # frame = cv2.resize(frame, (frame.shape[1]*2, frame.shape[0]*2))
        ret, thresh = cv2.threshold(frame, 30, 255, cv2.THRESH_BINARY_INV)
        thresh = cv2.bitwise_not(thresh)

        final = np.zeros(thresh.shape)
        final.fill(255)

        eye_black_area = np.argwhere(thresh == 0)
        pupil_position = np.average(eye_black_area[:,:2], axis=0)
        pupil_position = [round(x) for x in pupil_position]

        final[pupil_position[0], pupil_position[1]] = 0

        converted_x = range_convert(pupil_position[1], 0, 160, 0, display_width)
        converted_y = range_convert(pupil_position[0], 0, 90, 0, display_height)
        
        frame = cv2.circle(frame, (pupil_position[0],pupil_position[1]), radius=10, color=(0,0,255), thickness=-1)
        cv2.imshow('output', frame)
        cv2.waitKey(1)

        return (converted_x, converted_y)
        # return (pupil_position[1], pupil_position[0])
    except:
        return None

def move_mouse():

    try:
        f = open("ref.txt", "r")
    except FileNotFoundError: # if the file is not found, run calibration
        calibration(source)
    else:
        f.close()

    with open("ref.txt", "r") as f:
        top_left = list(map(int, f.readline().strip().split(',')))
        bottom_right = list(map(int, f.readline().strip().split(',')))


    cap = cv2.VideoCapture(source)
    if not cap.isOpened:
        print('Error opening video capture')
        exit(0)
    while True:
        ret, frame = cap.read()
        # frame = cv2.resize(frame,(1280,720))
        frame = cv2.flip(frame, 1)
        if frame is None:
            pass
        
        if cv2.waitKey(1) & 0xFF == ord('q' or 'Q'):
            cap.release()
            cv2.destroyAllWindows()
            break

        
        coords = left_pupil_track(frame)
        if coords is not None:
            # converted_x = range_convert(coords[0], 0, 160, top_left[0], bottom_right[0])
            converted_x = range_convert(coords[0], top_left[0], bottom_right[0], 0,  display_width)
            # converted_y = range_convert(coords[1], 0, 90, top_left[1], bottom_right[1])
            converted_y = range_convert(coords[1], top_left[1], bottom_right[1], 0, display_height)
            print(converted_x, converted_y)
            pyautogui.moveTo(converted_x, converted_y)


def calibration(source):

    top_left_coords = [0,0]
    bottom_right_coords = [0,0]

    def calib_reset():
        calibrate_base = np.zeros((display_height, display_width, 3))
        calibrate_base.fill(0)
        text_width, text_height = cv2.getTextSize("Calibration", cv2.FONT_HERSHEY_COMPLEX, 1, 2)[0]
        CenterCoordinates = (int(calibrate_base.shape[1] / 2)-int(text_width / 2), int(calibrate_base.shape[0] / 2) - int(text_height / 2))
        calibrate_base = cv2.putText(calibrate_base, "Calibration", CenterCoordinates, cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255))
        text_width, text_height = cv2.getTextSize("Please look at the crosshair", cv2.FONT_HERSHEY_COMPLEX, 1, 2)[0]
        CenterCoordinates = (int(calibrate_base.shape[1] / 2)-int(text_width / 2), int(calibrate_base.shape[0] / 2) - int(text_height / 2))
        calibrate_base = cv2.putText(calibrate_base, "Please look at the crosshair", (CenterCoordinates[0], CenterCoordinates[1]+text_height+10), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255))
        return calibrate_base

    cap = cv2.VideoCapture(source)
    
    # top left
    start = time()
    calib_flag = False
    counter = 0

    while (time() - start <= 5):
        if not calib_flag:
            calib_screen = calib_reset()
            cv2.circle(calib_screen, (50,50), 50, (0,0,255), 2)
            cv2.circle(calib_screen, (50,50), 5, (0,0,255), -1)
            cv2.namedWindow("Calibrate", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Calibrate", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("Calibrate", calib_screen)
            cv2.waitKey(1)
            calib_flag = True

        ret, frame = cap.read()

        coords = left_pupil_track(frame)
        if coords is not None:
            counter += 1
            top_left_coords[0] += coords[0]
            top_left_coords[1] += coords[1]
    
    cv2.destroyAllWindows()
    top_left_coords = [round(x/counter) for x in top_left_coords]
    # print(top_left_coords)

    # bottom right
    start = time()
    calib_flag = False
    counter = 0

    while (time() - start <= 5):
        if not calib_flag:
            calib_screen = calib_reset()
            cv2.circle(calib_screen, (display_width-50, display_height-50), 50, (0,0,255), 2)
            cv2.circle(calib_screen, (display_width-50, display_height-50), 5, (0,0,255), -1)
            cv2.namedWindow("Calibrate", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Calibrate", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("Calibrate", calib_screen)
            cv2.waitKey(1)
            calib_flag = True

        ret, frame = cap.read()

        coords = left_pupil_track(frame)
        if coords is not None:
            counter += 1
            bottom_right_coords[0] += coords[0]
            bottom_right_coords[1] += coords[1]

    cv2.destroyAllWindows()
    bottom_right_coords = [round(x/counter) for x in bottom_right_coords]
    # print(bottom_right_coords)

    with open("ref.txt", "w") as f:
        f.write(str(top_left_coords[0]) + "," + str(top_left_coords[1]))
        f.write("\n")
        f.write(str(bottom_right_coords[0]) + "," + str(bottom_right_coords[1]))

    cap.release()
    return


if __name__ == "__main__":
    calibration(1)
    move_mouse()