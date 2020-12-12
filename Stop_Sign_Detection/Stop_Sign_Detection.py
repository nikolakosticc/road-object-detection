import cv2
from playsound import playsound
import numpy as np


stop_sign_xml = cv2.CascadeClassifier('stop_sign.xml') #Create stop sign classifier

cap = cv2.VideoCapture("Stop_Sign_Detection_Video.mp4") #Test video

while True:
    ret, img = cap.read() #Read the first frame
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Detect_Stop_Sign = stop_sign_xml.detectMultiScale(gray, 1.3, 5) #Pass the frame to our body classifier

    for(x,y,w,h) in Detect_Stop_Sign:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) #Draw a rectangle around detected stop sign
        print("Stop Sign Detected")

    cv2.imshow("Stop Sign Detection", img)
    key = cv2.waitKey(30)
    if key == ord('q'): #Close the app on press of the button 'q'
        cap.release()
        cv2.destroyAllWindows()
        break