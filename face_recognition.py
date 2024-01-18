import cv2 as cv
import numpy as np

haar_cascade = cv.CascadeClassifier(r"C:/Users/devar/Desktop/opencv/haar_face.xml")
people = ["ronaldo","devarsh"]
features = np.load('features.npy', allow_pickle=True)
labels = np.load('labels.npy')
face_recognizer = cv.face.LBPHFaceRecognizer.create()
face_recognizer.read('face_trained.yml')

camera = cv.VideoCapture(0)

while True:
    rect, frame = camera.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    face_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)

    for (x, y, w, h) in face_rect:
        faces_roi = gray[y:y + h, x:x + w]
        label, confidence = face_recognizer.predict(faces_roi)
        print(f'label={people[label]} with a confidence of {confidence}')
        cv.putText(frame, str(people[label]), (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv.imshow('screen', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
camera.release()
cv.destroyAllWindows()
