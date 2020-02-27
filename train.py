import os
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from PIL import ImageFile
from model import define_model
import generators
import config
import argparse
import lr_scheduler
import sys

ImageFile.LOAD_TRUNCATED_IMAGES = True


# training_generator = training_generator(config.TRAINING_DATA_DIR)
# validation_generator = validation_generator(config.VALIDATION_DATA_DIR)
# ckpt_path = os.path.join(config.PATH, 'ckpt')


def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--model', type=tf.keras.Model, default=model.define_model(), help='what type of model and ')
    parser.add_argument('--lrs', type=str, default=None, help='defines the learning rate scheduler')
    parser.add_argument('--ckpt_path', type=str, default=os.path.join(config.PATH, 'ckpt'),
                        help='where to save the trained models checkpoints')
    args = parser.parse_args()
    train(args)


def train(args):
    training_generator = generators.training_generator(config.TRAINING_DATA_DIR)
    validation_generator = generators.validation_generator(config.TRAINING_DATA_DIR, config.VALIDATION_DATA_DIR)
    modelCheckpoint = ModelCheckpoint(os.path.join(args.ckpt_path, config.MODEL_FILE_NAME),
                                      monitor='val_acc',
                                      verbose=1, save_best_only=True, mode='max', period=10)
    model = define_model()
    #schedule = lr_scheduler.StepDecay(initAlpha=0.001, factor=0.25, dropEvery=10)
    opt = SGD(lr=0.001, momentum=0.9, decay=0.0)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=opt)
    if args.lrs == 'stepdecay':
        schedule = lr_scheduler.StepDecay(initAlpha=0.001, factor=0.25, dropEvery=10)
        model.fit_generator(training_generator, steps_per_epoch=len(training_generator.filenames) // config.BATCH_SIZE,
                             epochs=config.EPOCHS, validation_data=validation_generator,
                             validation_steps=len(validation_generator.filenames) // config.BATCH_SIZE,
                             callbacks=[modelCheckpoint, LearningRateScheduler(schedule)], verbose=2)

    elif args.lrs == 'linear':
        linear_schedule = lr_scheduler.PolinomialDecay(maxEpochs=config.EPOCHS, initAlpha=0.001, power=1)
        model.fit_generator(training_generator,
                                 steps_per_epoch=len(training_generator.filenames) // config.BATCH_SIZE,
                                 epochs=config.EPOCHS, validation_data=validation_generator,
                                 validation_steps=len(validation_generator.filenames) // config.BATCH_SIZE,
                                 callbacks=[modelCheckpoint, LearningRateScheduler(linear_schedule)], verbose=2)

    elif args.lrs == 'polinomial':
        polinomial_schedule = PolinomialDecay(maxEpochs=config.EPOCHS, initAlpha=0.001, power=5)
        model.fit_generator(training_generator,
                                 steps_per_epoch=len(training_generator.filenames) // config.BATCH_SIZE,
                                 epochs=config.EPOCHS, validation_data=validation_generator,
                                 validation_steps=len(validation_generator.filenames) // config.BATCH_SIZE,
                                 callbacks=[modelCheckpoint, LearningRateScheduler(polinomial_schedule)], verbose=2)


if __name__ == '__main__':
    main()
