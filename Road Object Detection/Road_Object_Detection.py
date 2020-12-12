import cv2
import numpy as np


body_classifier = cv2.CascadeClassifier('body.xml') #Create pedestrian classifier
stop_sign_classifier = cv2.CascadeClassifier('stop_sign.xml') #Create stop sign classifier
cap = cv2.VideoCapture('Test_Video_1.mp4') #Test Video


while cap.isOpened(): #Loop once video is successfully loaded

    ret, frame = cap.read() # Read first frame
    frame = cv2.resize(frame, None,fx=0.4, fy=0.4, interpolation = cv2.INTER_LINEAR) #Resize image to smaller quality in order to get bigger fps

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bodies = body_classifier.detectMultiScale(gray, 1.5, 3) #Pass frame to body classifier
    stop = stop_sign_classifier.detectMultiScale(gray, 1.5, 3) #Pass frame to stop sign classifier
    
    #Display detected pedestrians

    for (x,y,w,h) in bodies:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, 'Pedestrian Detected', (10,50), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA) #Pedestirans detected text


    #Display detected stop sign

    for (x,y,w,h) in stop:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (22, 31, 217), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, 'Stop Sign Detected', (10,100), font, 0.7, (22, 31, 217), 2, cv2.LINE_AA) #Stop sign detected text

    cv2.imshow('Road Object Detection', frame)
    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()