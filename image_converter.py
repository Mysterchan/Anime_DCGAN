import cv2
import matplotlib.pyplot as plt
import os
SIZE = (64,64)
PATH = "C:\\Users\\User\\Desktop\\GAN_Anime\\images"

for _,__,files in os.walk('C:\\Users\\User\\Desktop\\GAN_Anime\\images'):
    for file in files:
        file = os.path.join(PATH,file)
        image = cv2.imread(file)
        image = cv2.resize(image,SIZE)
        cv2.imwrite(file,image)