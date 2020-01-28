import os
import tensorflow as tf
from model.model import define_model, config
from model.utils import check_image_with_pil
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# training_generator = training_generator(config.TRAINING_DATA_DIR)
# validation_generator = validation_generator(config.VALIDATION_DATA_DIR)
ckpt_path = os.path.join(config.PATH, 'ckpt')


def train(ckpt_path:str):
    check_image_with_pil(config.TRAINING_DATA_DIR)
    check_image_with_pil(config.VALIDATION_DATA_DIR)
    training_generator = training_generator(config.TRAINING_DATA_DIR)
    validation_generator = validation_generator(config.VALIDATION_DATA_DIR)
    modelCheckpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(ckpt_path, config.MODEL_FILE_NAME), monitor='val_acc',
                                                         verbose=1, save_best_only=True, mode='max', period=10)
    model = define_model()
    model.fit_generator(training_generator, steps_per_epoch=len(training_generator.filenames) // BATCH_SIZE,
                        epochs=EPOCHS, validation_data=validation_generator,
                        validation_steps=len(validation_generator.filenames) // BATCH_SIZE,
                        callbacks=[ModelCheckpoint], verbose=2)


if __name__ == '__main__':
    train(ckpt_path=ckpt_path)
