import os
from unittest import result
import cv2
import utils
import numpy as np
import matplotlib.pyplot as plt



def detect(dataPath, clf):
    """
    Please read detectData.txt to understand the format. Load the image and get
    the face images. Transfer the face images to 19 x 19 and grayscale images.
    Use clf.classify() function to detect faces. Show face detection results.
    If the result is True, draw the green box on the image. Otherwise, draw
    the red box on the image.
      Parameters:A
        dataPath: the path of detectData.txt
      Returns:
        No returns.
    """
    # Begin your code (Part 4)
    '''
    This function loads the image paths and bounding box information from a text file specified by 'dataPath'.
    It then iterates over each image, reads it, converts it to grayscale, and crops the regions specified by the bounding box.
    For each cropped region, it resizes it to 19x19 and converts it to grayscale.
    It then uses the provided classifier 'clf' to classify whether the region contains a face or not.
    If the classification result is True, it draws a green rectangle around the face; otherwise, it draws a red rectangle.
    The function displays the image with the drawn rectangles.
    '''
    folderPath='data/detect/'
    with open(dataPath, 'r') as file:
        text=file.readlines()
        while len(text):
            name, num=text[0].split()
            text.pop(0)
            img=cv2.imread(folderPath+name)
            gray_img=cv2.imread(folderPath+name, cv2.IMREAD_GRAYSCALE)
            for i in range(int(num)):
                x0, y0, x1, y1=map(int, text[0].split())
                x1+=x0
                y1+=y0
                text.pop(0)
                if clf.classify(cv2.resize(gray_img[y0:y1, x0:x1], (19, 19))):
                    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), thickness=3)
                else:
                    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), thickness=3)
            cv2.imshow('image', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    # End your code (Part 4)
