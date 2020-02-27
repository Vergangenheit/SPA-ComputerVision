import tensorflow as tf
import config
import os


def training_generator(training_data_dir):
    # Data augmentation
    training_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True)

    training_gen = training_data_generator.flow_from_directory(
        training_data_dir,
        target_size=(config.IMAGE_WIDTH, config.IMAGE_HEIGHT),
        batch_size=config.BATCH_SIZE,
        classes=os.listdir(training_data_dir))

    return training_gen


def validation_generator(training_data_dir, validation_data_dir):
    validation_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

    validation_gen = validation_data_generator.flow_from_directory(
        validation_data_dir,
        target_size=(config.IMAGE_WIDTH, config.IMAGE_HEIGHT),
        batch_size=config.BATCH_SIZE,
        classes=os.listdir(training_data_dir))

    return validation_gen


def test_generator(test_data_dir):
    test_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

    test_gen = test_data_generator.flow_from_directory(test_data_dir,
                                                       target_size=(config.IMAGE_WIDTH, config.IMAGE_HEIGHT),
                                                       batch_size=1, class_mode=None, shuffle=True)

    return test_gen
