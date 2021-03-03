from PIL import Image
import pytesseract
import numpy as np 
import cv2
import pandas
img = cv2.imread("input.jpg")
cv2.imshow('img1', img)
cv2.waitKey(1000)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('img2', img)
img = cv2.resize(img, None, fx=1, fy=1, interpolation=cv2.INTER_CUBIC)
img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
cv2.imshow('img3', img)
kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
img = cv2.filter2D(img, -1, kernel)
mser = cv2.MSER_create()
vis = img.copy()
regions, _ = mser.detectRegions(img)
hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
cv2.polylines(vis, hulls, 1, (0,255, 0))
cv2.imshow('hulls', vis)
cv2.waitKey(7000)
mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)

for contour in hulls:

    cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)


mask = cv2.bitwise_not(mask)
dimensions = img.shape
height = img.shape[0]
width = img.shape[1]
contours,hierarchy = cv2.findContours(mask, 1, 2)

for c in contours:
    rect = cv2.boundingRect(c)
    if rect[2] > (height/3) and rect[3] > (width/3):
        continue
    x,y,w,h = rect
    cv2.rectangle(mask,(x-(width//80),y-(height//80)),(x+w+(width//80),y+h+(height//80)),(0,255,0), cv2.FILLED) 
cv2.imwrite("edited.png", img)
cv2.imwrite("mask.png", mask)
cv2.imshow('mask', mask)
cv2.waitKey(7000)
img = cv2.bitwise_or(img, mask) 
cv2.imwrite("final.png", img)
cv2.imshow('final', img)
cv2.waitKey(7000)
cv2.destroyAllWindows()
text = pytesseract.image_to_data(Image.open("final.png"), output_type='data.frame')
text = text[text.conf >80]
print(text) 
lines = text.groupby('block_num')['text'].apply(list)
block = text.groupby('page_num')['block_num'].apply(list)
block[1] =list(dict.fromkeys(block[1]))
for x in block[1]:
    print(*lines[x])

