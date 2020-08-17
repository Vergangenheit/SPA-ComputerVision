import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from random import shuffle
import cv2
import os
from configs import config


def load_model(saved_model: str):
    model = tf.keras.models.load_model(saved_model)

    return model

def extract_test_sample(samples=10):
    data = (os.listdir(config.TEST_DATA_DIR))
    shuffle(data)
    sample = data[:samples]

    return sample

def predict_and_visualize(model, sample):
    for image in sample:
        img = cv2.imread(os.path.join(config.TEST_DATA_DIR, image))
        im = mpimg.imread(os.path.join(config.TEST_DATA_DIR, image))

        x = cv2.resize(img, (config.IMAGE_SIZE, config.IMAGE_SIZE))
        x = x / 255.
        x = np.expand_dims(x, axis=0)
        preds = model.predict(x)
        plt.imshow(im)
        if np.argmax(preds[0]) == 0:
            plt.title("Image is %s but prediction is %.2f" % (image.split('_')[0], (preds[0][0]) * 100) + ' ' +
                      os.listdir(config.TRAINING_DATA_DIR)[0])
        elif np.argmax(preds[0]) == 1:
            plt.title("Image is %s but prediction is %.2f" % (image.split('_')[0], (preds[0][1]) * 100) + ' ' +
                      os.listdir(config.TRAINING_DATA_DIR)[1])
        elif np.argmax(preds[0]) == 2:
            plt.title("Image is %s but prediction is %.2f" % (image.split('_')[0], (preds[0][2]) * 100) + ' ' +
                      os.listdir(config.TRAINING_DATA_DIR)[2])
        elif np.argmax(preds[0]) == 3:
            plt.title("Image is %s but prediction is %.2f" % (image.split('_')[0], (preds[0][3]) * 100) + ' ' +
                      os.listdir(config.TRAINING_DATA_DIR)[3])
        elif np.argmax(preds[0]) == 4:
            plt.title("Image is %s but prediction is %.2f" % (image.split('_')[0], (preds[0][4]) * 100) + ' ' +
                      os.listdir(config.TRAINING_DATA_DIR)[4])
        elif np.argmax(preds[0]) == 5:
            plt.title("Image is %s but prediction is %.2f" % (image.split('_')[0], (preds[0][5]) * 100) + ' ' +
                      os.listdir(config.TRAINING_DATA_DIR)[5])
        plt.show()





