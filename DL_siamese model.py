import os
import csv
def image_store(x):
# Define the starting directory
    start_directory = x
    # Create a list to store the file paths and their respective directory names
    file_paths = []

    # Walk through the directory and collect file paths
    for root, dirs, files in os.walk(start_directory):
        for file_name in files:
            # Construct the full file path
            file_path = os.path.join(root, file_name)

            # Store the file path along with the directory name
            directory_name = os.path.basename(root)
            file_paths.append((directory_name, file_path))

    # Define the CSV file name
    csv_file_name = "file_paths.csv"

    # Write the file paths to a CSV file
    with open(csv_file_name, mode="w", newline="") as file:
        writer = csv.writer(file)

        # Write the header row
        writer.writerow(["Directory", "File Path"])

        # Write the file paths 
        for directory, file_path in file_paths:
            writer.writerow([directory, file_path])

    print(f"File paths have been saved to {csv_file_name}.")
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2 as cv
import numpy as np
def train_test_spliting(x):
    df=pd.read_csv(x)
    labels=df['Directory']
    features=df['File Path']
    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)
    return X_train,X_val,y_train,y_val

def train_network(X_train,X_val,y_train,y_val):
    img_list_train=[]
    img_list_val=[]
    for i in X_train:
        img=cv.imread(i)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = cv.resize(img, (128,128))
        img_array=np.array(img)
        img_array=img_array/255.0
        img_list_train.append(img_array)
    image_arrays_train=np.array(img_list_train)
    image_arrays_train = image_arrays_train.reshape((-1, 128, 128, 1))
    y_train_np = np.array(y_train)
    assert y_train_np.shape[0] == image_arrays_train.shape[0]
    # # Get the number of samples in the training data
    # num_samples = len(image_arrays_train)
    # # Reshape y_train_np to have the same shape as image_arrays
    # y_train_np = y_train_np.reshape((num_samples, 128, 128))


    for t in X_val:
        img=cv.imread(t)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = cv.resize(img, (128,128))
        img_array=np.array(img)
        img_array=img_array/255.0
        img_list_val.append(img_array)
    image_arrays_val = np.array(img_list_val)
    image_arrays_val = image_arrays_val.reshape((-1, 128, 128, 1))
    y_val_np = np.array(y_val)
    assert y_val_np.shape[0] == image_arrays_val.shape[0]
    # # Get the number of samples in the training data
    # num_samples = len(image_arrays_val)
    # # Reshape y_train_np to have the same shape as image_arrays
    # y_val_np = y_train_np.reshape((num_samples, 128, 128))

    return image_arrays_train, y_train_np, image_arrays_val, y_val_np

# def create_pairs(data, labels):
#     pairs = []
#     labels = np.array(labels)
#     num_classes = max(labels.unique().count()) + 1
#
#     for class_idx in range(num_classes):
#         class_indices = np.where(labels == class_idx)[0]
#         for i in range(len(class_indices)):
#             idx1 = class_indices[i]
#             idx2 = class_indices[(i + 1) % len(class_indices)]  # Ensure a dissimilar pair
#             pairs.append([data[idx1], data[idx2], 1])  # Similar pair
#
#         for _ in range(len(class_indices)):
#             idx1 = class_indices[np.random.randint(0, len(class_indices))]
#             idx2 = class_indices[np.random.randint(0, len(class_indices))]
#             while idx2 == idx1:  # Ensure a dissimilar pair
#                 idx2 = class_indices[np.random.randint(0, len(class_indices))]
#             pairs.append([data[idx1], data[idx2], 0])  # Dissimilar pair
#
#     return np.array(pairs)
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Lambda
from tensorflow.keras.models import Model
def create_siamese_network():
    input_shape=(128,128)
    # Define the input layers for the two images
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    # Define the CNN model that will be shared between the two inputs
    base_network = tf.keras.Sequential([
        Conv2D(64, (10, 10), activation='relu', input_shape=input_shape),
        Flatten(),
        Dense(128, activation='relu'),
    ])

    # Process each input through the shared network
    encoded_a = base_network(input_a)
    encoded_b = base_network(input_b)

    # Calculate the Euclidean distance between the two encodings
    distance = Lambda(lambda x: tf.keras.backend.abs(x[0] - x[1]))([encoded_a, encoded_b])

    # Create the Siamese Network model
    siamese_network = Model(inputs=[input_a, input_b], outputs=distance)

    return siamese_network

if __name__=="__main__":
    x="C:/Users/devar/Desktop/python project/faces/myphotos"
    image_store(x)
    y='file_paths.csv'
    X_train, X_val, y_train, y_val=train_test_spliting(y)
    # print(X_train)
    image_arrays_train, y_train_np, image_arrays_val, y_val_np=train_network(X_train,X_val,y_train,y_val)
    # image_arrays.shape
    # print(image_arrays.shape)

    siamese_model = create_siamese_network()

    # Compile the model (you can adjust the loss and optimizer as needed)
    siamese_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Print a summary of the model
    siamese_model.summary()

    epochs = 4
    steps_per_epoch = len(X_train) // 5
    validation_steps = len(X_val) // 5

    history = siamese_model.fit(image_arrays_train,y_train_np,
                        epochs=epochs,
                        # steps_per_epoch=steps_per_epoch,
                        validation_data=(image_arrays_val, y_val_np))