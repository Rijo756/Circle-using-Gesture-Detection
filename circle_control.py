import cv2
import numpy as np
import time
import mediapipe as mp
import math

#Defining Webcam as video input
cap = cv2.VideoCapture(1)
ptime = 0
ctime = 0

#defining hand module
mp_Hand = mp.solutions.hands
hands = mp_Hand.Hands() #static_image_mode=False, max_num_hands=2, min_detection_confidence = 0.5,min_tracking_confidence=0.5
mp_draw = mp.solutions.drawing_utils



def drawcircle(screen,radius = 30,centerx=500,centery=320):
    cv2.circle(screen,(centerx,centery), radius , (0,255,0), -1)


while True:
    Screen = np.zeros((720,1280,3),np.uint8)
    success,img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    distance = 15
    centerx=500
    centery=320
    #getting the hands from the image using media
    results = hands.process(imgRGB)

    image_height, image_width, _ = img.shape

    #using mediapipe module to draw the landmarks into image
    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:   
            mp_draw.draw_landmarks(img,handlms, mp_Hand.HAND_CONNECTIONS)
            #getting thumb position
            x4,y4 = handlms.landmark[4].x,handlms.landmark[4].y
            x4 = int(x4 * image_height)
            y4 = int(y4 * image_width)
            #getting index finger position
            x8,y8 = handlms.landmark[8].x,handlms.landmark[8].y
            x8 = int(x8 * image_height)
            y8 = int(y8 * image_width) 
            distance = int(math.sqrt(pow(x8-x4,2)+pow(y8-y4,2)))
            centerx = int((x4+x8)* Screen.shape[1]/ (2*image_height))
            centery = int((y4+y8)* Screen.shape[0]/ (2*image_width))
            #print (distance)

    scale_percent = 50 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
      
    # resize image
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    x,y,z = img.shape
    Screen [-x:,-y:,:] = img
    #to calculate fps
    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime
    cv2.putText(Screen,"FPS: "+str(round(fps)),(20,20),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1)

    radius = abs(distance-15)
    drawcircle(Screen,radius,centerx, centery)
    
    cv2.imshow("CAM001",Screen)
    cv2.waitKey(1)
