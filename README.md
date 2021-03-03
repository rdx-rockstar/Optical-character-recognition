# Optical-character-recognition

This is a python project to read charecter from the input image.
The input image is scanned for all contours using mser and creates smallest bounding rectanglle to create mask so as the limit the search area of ocr.  
Then using the ocr Tesseract we scan the masked image to detect text in the modifies image.
