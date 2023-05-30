import cv2 as cv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
img = cv.imread('sample_images/ImagesOriginalSize/Scol/N93, Rt TL AIS, F, 14 Yrs.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"

C = 6
AR = 5

#img = cv.medianBlur(img,5)
ret,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
            cv.THRESH_BINARY_INV,AR,C)
th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY_INV,AR,C)

edges1 = cv.Canny(img,50,100)
edges2 = cv.Canny(th2,100,200)
edges3 = cv.Canny(th3,100,200)

df = pd.DataFrame(edges1).replace(255, 1)
titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding', 'Canny Original', 'Canny MEAN', 'Canny GAUSSIAN', 'MEANS']
images = [img, th1, th2, th3, edges1, edges2, edges3]
for i in range(len(images)):
    plt.subplot(2,4,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()
