import cv2 as cv

xml_path = 'C:/Users/devar/Desktop/opencv/haar_face.xml'  # Provide the correct path to your XML cascade file

try:
    img = cv.imread(r"C:/Users/devar/Desktop/faces/myphotos/Devarsh/IMG-20230806-WA0012.jpg")
    cv.imshow('my', img)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow('Gray People', gray)

    haar_cascade = cv.CascadeClassifier(xml_path)

    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)

    print(f'Number of faces found = {len(faces_rect)}')

    for (x, y, w, h) in faces_rect:
        print(f'Face found at: ({x}, {y}), Width: {w}, Height: {h}')
        cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=5)

    cv.imshow('Detected Faces', img)

    cv.waitKey(0)
    cv.destroyAllWindows()

except Exception as e:
    print(f"An error occurred: {e}")
    cv.waitKey(0)
