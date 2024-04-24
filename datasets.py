import numpy as np
import os
import cv2

def load_custom_dataset(data_path='./data/custom_dataset'):
    x = []
    y = []
    for label in os.listdir(data_path):
        label_path = os.path.join(data_path, label)
        if os.path.isdir(label_path):
            for image_file in os.listdir(label_path):
                image_path = os.path.join(label_path, image_file)
                # Read image
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                # Resize image to 28x28 if necessary
                img = cv2.resize(img, (28, 28))
                # Normalize pixel values to [0, 1]
                img = img.astype('float32') / 255.
                # Append image and label to lists
                x.append(img)
                y.append(int(label))
    x = np.array(x)
    y = np.array(y)
    print('Custom dataset:', x.shape, y.shape)
    return x, y


