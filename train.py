import os
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from PIL import ImageFile
import model
import generators
import config
import argparse
import sys

ImageFile.LOAD_TRUNCATED_IMAGES = True


# training_generator = training_generator(config.TRAINING_DATA_DIR)
# validation_generator = validation_generator(config.VALIDATION_DATA_DIR)
# ckpt_path = os.path.join(config.PATH, 'ckpt')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=tf.keras.Model, default=model.define_model(), help='what type of model and ')
    parser.add_argument('-lrs', type=LearningRateScheduler, default=None, help='defines the learning rate scheduler')
    parser.add_argument('--ckpt_path', type=str, default=os.path.join(config.PATH, 'ckpt'),
                        help='where to save the trained models checkpoints')
    args = parser.parse_args()
    sys.stdout.write(str(train(args)))


def train(args):
    training_generator = generators.training_generator(config.TRAINING_DATA_DIR)
    validation_generator = generators.validation_generator(config.VALIDATION_DATA_DIR)
    modelCheckpoint = ModelCheckpoint(os.path.join(args.ckpt_path, config.MODEL_FILE_NAME),
                                      monitor='val_acc',
                                      verbose=1, save_best_only=True, mode='max', period=10)
    #args.model = define_model()
    #schedule = lr_scheduler.StepDecay(initAlpha=0.001, factor=0.25, dropEvery=10)
    opt = SGD(lr=0.001, momentum=0.9, decay=0.0)
    args.model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=opt)
    args.model.fit_generator(training_generator, steps_per_epoch=len(training_generator.filenames) // config.BATCH_SIZE,
                             epochs=config.EPOCHS, validation_data=validation_generator,
                             validation_steps=len(validation_generator.filenames) // config.BATCH_SIZE,
                             callbacks=[modelCheckpoint, LearningRateScheduler(args.lrs)], verbose=2)


if __name__ == '__main__':
    main()
