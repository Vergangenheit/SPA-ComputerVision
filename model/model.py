from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from configs.config import CFG
from model.base_model import BaseModel
import os
from model import lr_scheduler


class VGGSPA(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.base_model = VGG16(include_top=False, input_shape=self.config.model.input)
        self.batch_size = self.config.train.batch_size
        self.model = None
        self.opt = None
        self.epochs = self.config.train.epochs
        self.steps_per_epoch = 0

        self.image_size = self.config.train.image_size
        self.training_generator = None
        self.validation_generator = None
        self.test_generator = None

    def training_generator(self):
        # Data augmentation
        training_data_generator = ImageDataGenerator(
            rescale=self.config.train.preprocess.rescale,
            shear_range=self.config.train.preprocess.shear_range,
            zoom_range=self.config.train.preprocess.zoom_range,
            horizontal_flip=self.config.train.preprocess.horizontal_flip)

        self.training_generator = training_data_generator.flow_from_directory(
            self.config.data.path.train_path,
            target_size=(self.config.train.image_size, self.config.train.image_size),
            batch_size=self.batch_size,
            classes=os.listdir(self.config.data.path.train_path))

    def validation_generator(self):
        validation_data_generator = ImageDataGenerator(rescale=self.config.train.preprocess.rescale)

        self.validation_generator = validation_data_generator.flow_from_directory(
            self.config.data.path.valid_path,
            target_size=(self.config.train.image_size, self.config.train.image_size),
            batch_size=self.batch_size,
            classes=os.listdir(self.config.data.path.valid_path))

    def test_generator(self):
        test_data_generator = ImageDataGenerator(rescale=self.config.train.preprocess.rescale)

        self.test_generator = test_data_generator.flow_from_directory(self.config.data.path.test_path,
                                                                      target_size=(self.config.train.image_size,
                                                                                   self.config.train.image_size),
                                                                      batch_size=1, class_mode=None, shuffle=True)

    def build(self):
        for layer in self.base_model.layers:
            layer.trainable = False

        # add new classifier layers
        flatten1 = Flatten()(self.base_model.layers[-1].output)
        class1 = Dense(self.config.model.dense1.units, activation=self.config.model.dense1.activation,
                       kernel_initializer=self.config.model.dense1.kernel_initializer)(flatten1)
        output = Dense(self.config.model.output.units, activation=self.config.model.output.activation)(class1)
        # define new model
        self.model = Model(input=self.base_model.inputs, outputs=output)

    def train(self, args):
        self.opt = SGD(lr=self.config.train.optimizer.lr, momentum=self.config.train.optimizer.momentum)
        self.model.compile(loss=self.config.train.loss, metrics=self.config.train.metrics, optimizer=self.opt)
        modelCheckpoint = ModelCheckpoint(os.path.join(args.ckpt_path, self.config.model.output.model_filename),
                                          monitor='val_acc',
                                          verbose=1, save_best_only=True, mode='max', period=10)
        if args.lrs == 'stepdecay':
            schedule = lr_scheduler.StepDecay(initAlpha=0.001, factor=0.25, dropEvery=10)
            model_history = self.model.fit_generator(self.training_generator, steps_per_epoch=len(
                self.training_generator.filenames) // self.batch_size,
                                     epochs=self.epochs, validation_data=self.validation_generator,
                                     validation_steps=len(
                                         self.validation_generator.filenames) // self.batch_size,
                                     callbacks=[modelCheckpoint, LearningRateScheduler(schedule)],
                                     verbose=self.config.train.verbose)

        elif args.lrs == 'linear':
            linear_schedule = lr_scheduler.PolinomialDecay(maxEpochs=self.epochs, initAlpha=0.001, power=1)
            model_history = self.model.fit_generator(self.training_generator, steps_per_epoch=len(
                self.training_generator.filenames) // self.batch_size,
                                     epochs=self.epochs, validation_data=self.validation_generator,
                                     validation_steps=len(
                                         self.validation_generator.filenames) // self.batch_size,
                                     callbacks=[modelCheckpoint, LearningRateScheduler(linear_schedule)],
                                     verbose=self.config.train.verbose)

        elif args.lrs == 'polinomial':
            polinomial_schedule = lr_scheduler.PolinomialDecay(maxEpochs=self.epochs, initAlpha=0.001, power=5)

            model_history = self.model.fit_generator(self.training_generator, steps_per_epoch=len(
                self.training_generator.filenames) // self.batch_size,
                                                     epochs=self.epochs, validation_data=self.validation_generator,
                                                     validation_steps=len(
                                                         self.validation_generator.filenames) // self.batch_size,
                                                     callbacks=[modelCheckpoint, LearningRateScheduler(polinomial_schedule)],
                                                     verbose=self.config.train.verbose)

        return model_history.history['loss'], model_history.history['val_loss']

    def evaluate(self):
        pass

