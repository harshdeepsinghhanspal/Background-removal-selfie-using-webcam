import cvzone
import cv2
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(cv2.CAP_PROP_FPS, 60) #60FPS

segmentor = SelfiSegmentation()
fpsReader = cvzone.FPS()

listIMG = os.listdir('Resources')
imgList = []
for imgPath in listIMG:
    img = cv2.imread(f'Resources/{imgPath}')
    imgList.append(img)

indexImg = 0

while True:
    success, img = cap.read()
    imgOut = segmentor.removeBG(img, imgList[indexImg], threshold=0.8)
    #If one, then everything is out
    #Just replace imgBG with (b,g,r) color combo for simple colours

    imgStack = cvzone.stackImages([img, imgOut],2,1)
    _, imgStack = fpsReader.update(imgStack, color=(0,0,255))

    cv2.imshow('Result Combn', imgStack)
    key = cv2.waitKey(1)
    if key == ord('a'):
        if indexImg>0:
            indexImg -= 1

    elif key == ord('d'):
        if indexImg < len(imgList)-1:
            indexImg += 1

    elif key == ord('q'):
        break