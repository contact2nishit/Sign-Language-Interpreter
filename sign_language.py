import cv2
import mediapipe as mp
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)

offset = 30
imgSize = 300
counter = 0

folder = "Data/FuckYou"

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    # Cropping the actual image:
    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']
        
        # Create a fixed size white image matrix:
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # imgCrop = img[y-offset:y + offset + h, x-offset:x + offset + w]
        imgCrop = img[max(0, y-offset):min(img.shape[0], y + offset + h), 
                       max(0, x-offset):min(img.shape[1], x + offset + w)]

        imgCropShape = imgCrop.shape

        # try:
        #     imgWhite[0:imgCropShape[0], 0:imgCropShape[1]] = imgCrop
        # except:
        #     pass

        
        ratio = h/w

        if ratio > 1:
            k = imgSize / h
            new_width = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop, (new_width, imgSize))
            imgResizeShape = imgCrop.shape
            wGap = math.ceil((imgSize-new_width)/2)
            
            try:
                imgWhite[:, wGap:new_width + wGap] = imgResize
            except:
                pass
        else:
            k = imgSize / w
            new_height = math.ceil(k*h)
            imgResize = cv2.resize(imgCrop, (imgSize, new_height))
            imgResizeShape = imgCrop.shape
            hGap = math.ceil((imgSize-new_height)/2)
            
            try:
                imgWhite[hGap:new_height + hGap, :] = imgResize
            except:
                pass




        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)

    key = cv2.waitKey(1)

    if key & 0XFF == ord('q'):
        break
    
    if key == ord("k"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)

    