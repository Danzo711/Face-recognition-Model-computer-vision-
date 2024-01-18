import cv2 as cv
#import pytesseract
import numpy as np
import pandas as pd
from pyzbar.pyzbar import decode


# Set the path to the Tesseract executable
#/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages (from pytesseract) (23.1)
#/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages (from pytesseract) (10.0.0)
# pytesseract.pytesseract.tesseract_cmd = r'/Users/devar/Desktop/opencv/New folder/tesseract.exe'

camera = cv.VideoCapture(0)
data = {"ID": [], "Text": []}
j = 0



user=[]
with open("names.txt","r") as file:
        names = file.read().split(",")
        # print(names)
        user=names
        print(user)
while True:
    ret, frame = camera.read()
    
 
    for i in user:
        # image = cv.imread(filename=f"{user}.png")
        #if frame is not None:
            decoded_objects = decode(frame)
           # print(decoded_objects)
            if decoded_objects:
                for obj in decoded_objects:
                    qr_data = obj.data.decode('utf-8')
                    #print(qr_data)
                    if qr_data in user:
                        print(f"QR code data for {qr_data}: {qr_data}")
                        print("\n access granted \n")
                    else:
                        print("access denied")
                        continue
            #else:
               # print(f"No QR codes found in {i}.png")
        #else:
            #print(f"Error loading image: {i}.png")

        # Select the document region
        # Define source and destination points for perspective transformation
    # src_pts = np.float32([[141, 131], [480, 159], [493, 630], [64, 601]])
    # dst_pts = np.float32([[0, 0], [800, 0], [800, 800], [0, 800]])
    #
    #     # Get perspective transformation matrix
    # M = cv.getPerspectiveTransform(src_pts, dst_pts)
    #
    #     # Convert to grayscale
    # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #
    #     # Apply Gaussian Blur
    # gray = cv.GaussianBlur(gray, (5, 5), 0)
    #
    #     # Apply adaptive thresholding
    # thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 25, 10)
    #
    #     # Apply morphological operations
    # kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    # thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
    #
    #     # Detect edges
    # edges = cv.Canny(thresh, 50, 150)
    # contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # if contours:
    #         largest_contour = max(contours, key=cv.contourArea)
    #         doc_region = cv.warpPerspective(thresh, M, (1000, 1000))
    #
    #         # Perform OCR using Tesseract
    #         text = pytesseract.image_to_string(doc_region)
    #
    #         # Display results
    #         cv.imshow('Original', frame)
    #         # cv.imshow('Thresholded', thresh)
    #         # cv.imshow('Edges', edges)
    #         # cv.imshow('Document Region', doc_region)
    #
    #         print(text)
    #
    #         data["Text"].append(text)
    #         data["ID"].append(j)
    #         j += 1

    if cv.waitKey(1) & 0xFF == ord('q'):
       break  # Press 'q' key to exit the loop
    

camera.release()
cv.destroyAllWindows()

# Save the data to a CSV file
# pd.DataFrame(data).to_csv(".\\text.csv", index=False)
# cv.waitKey(0)