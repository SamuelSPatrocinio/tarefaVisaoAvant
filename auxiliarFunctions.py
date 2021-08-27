import cv2
import numpy as np

def maskOutColor(image,lower,upper):
    LOWER_RANGE = np.array(lower)
    UPPER_RANGE = np.array(upper)
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv,LOWER_RANGE,UPPER_RANGE)

def floodFillandInvert(img):
    image = img.copy()
    height, width = image.shape[:2]
    mask = np.zeros((height+2, width+2), np.uint8)
    cv2.floodFill(image,mask,(0,0),255)
    return cv2.bitwise_not(image)

def centerImage(dimentions,dtype,color,image):
    bigImage = np.zeros(dimentions,dtype=dtype)
    bigImage.fill(color)
    center = int(bigImage.shape[0]/2)       
    top = int(center - image.shape[0]/2)
    bottom = int(center + image.shape[0]/2)
    left = int(center - image.shape[1]/2)
    right = int(center + image.shape[1]/2)
    bigImage[top:bottom,left:right] = image
    return bigImage