# import os
# import cv2 as cv
# import numpy as np

# people = ["myphotos"]

# dir = r"C:/Users/devar/Desktop/faces/myphotos"
# try:
#     haar_cascade = cv.CascadeClassifier("C:/Users/devar/Desktop/opencv/haar_face.xml")
# except FileNotFoundError:
#     print("Error loading Haar cascade classifier.")

# # Lists to store training data
# features = []
# labels = []

# def create_train():
#     for person in people:
#         path = os.path.join(dir, person)
#         label = people.index(person)

#         if not os.path.exists(path):
#             print(f"Path '{path}' does not exist.")
#             continue

#         for img in os.listdir(path):
#             img_path = os.path.join(path, img)
#             img_array = cv.imread(img_path)
#             if img_array is None:
#                 print(f"Error loading image: {img_path}")
#                 continue
#             gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
#             faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)
#             if len(faces_rect) == 0:
#                 print(f"No face detected in image: {img_path}")
#                 continue
#             for (x, y, w, h) in faces_rect:
#                 faces_roi = gray[y:y+h, x:x+w]
#                 features.append(faces_roi)
#                 labels.append(label)


# create_train()
# print('Training data collected.')
# print(f"print {features}",features.shape)
# print(f"print {labels}",labels.shape)

# features = np.array(features, dtype=object)
# labels = np.array(labels)

# # Create LBPH face recognizer
# face_recognizer = cv.face.LBPHFaceRecognizer.create()

# # Train the recognizer
# face_recognizer.train(features, labels)

# # Save the trained recognizer
# try:
#     face_recognizer.save('face_trained.yml')
#     print('Trained recognizer saved.')
# except cv.error as e:
#     print("Error saving trained recognizer:", e)

# # Save features and labels
# try:
#     np.save('features.npy', features)
#     np.save('labels.npy', labels)
#     print('Features and labels saved.')
# except Exception as e:
#     print("Error saving features and labels:", e)


import os
import cv2 as cv
import numpy as np

people = ["myphotos"]

# dir=r"C:/Users/devar/Desktop/opencv/faces/myphotos/ronaldo"
dir="C:/Users/devar/Desktop/opencv/text1.csv"

try:
    haar_cascade = cv.CascadeClassifier("C:/Users/devar/Desktop/opencv/haar_face.xml")
except FileNotFoundError:
    print("Error loading Haar cascade classifier.")

# Lists to store training data
features = []
labels = []

def create_train():
    for person in people:
        path = os.path.join(dir, person)
        label = people.index(person)

        if not os.path.exists(path):
            print(f"Path '{path}' does not exist.")
            continue

        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            img_array = cv.imread(img_path)
            if img_array is None:
                print(f"Error loading image: {img_path}")
                continue
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)
            if len(faces_rect) == 0:
                print(f"No face detected in image: {img_path}")
                continue
            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

create_train()
print('Training data collected.')
features = np.array(features, dtype=object)  # Convert to a NumPy array
labels = np.array(labels)
print(f"faces detected {features}")
print(f"faces detected {labels}")
print("Features shape:", features.shape)
print("Labels shape:", labels.shape)

# Create LBPH face recognizer
face_recognizer = cv.face.LBPHFaceRecognizer.create()

# Train the recognizer
face_recognizer.train(features, labels)

# Save the trained recognizer
try:
    face_recognizer.save('face_trained.yml')
    print('Trained recognizer saved.')
except cv.error as e:
    print("Error saving trained recognizer:", e)

# Save features and labels
try:
    np.save('features.npy', features)
    np.save('labels.npy', labels)
    print('Features and labels saved.')
except Exception as e:
    print("Error saving features and labels:", e)
