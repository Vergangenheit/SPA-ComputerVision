import os
import tensorflow as tf
from model.model import define_model
from model.utils import check_image_with_pil
from PIL import ImageFile
from model import generators, config

ImageFile.LOAD_TRUNCATED_IMAGES = True

# training_generator = training_generator(config.TRAINING_DATA_DIR)
# validation_generator = validation_generator(config.VALIDATION_DATA_DIR)
ckpt_path = os.path.join(config.PATH, 'ckpt')


def train(path:str):
    training_generator = generators.training_generator(config.TRAINING_DATA_DIR)
    validation_generator = generators.validation_generator(config.VALIDATION_DATA_DIR)
    modelCheckpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(ckpt_path, config.MODEL_FILE_NAME), monitor='val_acc',
                                                         verbose=1, save_best_only=True, mode='max', period=10)
    model = define_model()
    model.fit_generator(training_generator, steps_per_epoch=len(training_generator.filenames) // config.BATCH_SIZE,
                        epochs=EPOCHS, validation_data=validation_generator,
                        validation_steps=len(validation_generator.filenames) //config.BATCH_SIZE,
                        callbacks=[modelCheckpoint], verbose=2)


if __name__ == '__main__':
    train(path=ckpt_path)
