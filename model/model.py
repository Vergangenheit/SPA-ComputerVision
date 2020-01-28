from tf.keras.applications.vgg16 import VGG16
from tf.keras.models import Model
from tf.keras.layers import Flatten, Dense
from tf.keras.optimizers import SGD


def define_model():
    model = VGG16(include_top=False, input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3))
    for layer in model.layers:
        layer.trainable = False
    # add new classifier layers
    flatten1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flatten1)
    output = Dense(6, activation='softmax')(class1)
    # define new model
    model = Model(inputs=model.inputs, outputs=output)
    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=opt)

    return model
