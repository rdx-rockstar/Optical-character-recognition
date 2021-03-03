from PIL import Image
import cv2
import pytesseract
import numpy as np
img = cv2.imread("help.bmp")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, None, fx=1, fy=1, interpolation=cv2.INTER_CUBIC)
img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
img = cv2.filter2D(img, -1, kernel)
text=pytesseract.image_to_string(img)
print(text)

