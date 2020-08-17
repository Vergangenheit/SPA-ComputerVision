# Hyperparams
IMAGE_SIZE = 224
# IMAGE_WIDTH = 375
# IMAGE_HEIGHT = 500
IMAGE_WIDTH, IMAGE_HEIGHT = IMAGE_SIZE, IMAGE_SIZE
EPOCHS = 50
BATCH_SIZE = 25
TEST_SIZE = 30
INIT_LR = 1e-3
PATH = '/content/drive/My Drive/Trivago-SPA&Wellness'
MODEL_FILE_NAME = "VGG16_Wellness-epoch{epoch:03d}-loss{loss:.4f}-acc{acc:.4f}-val_loss{val_loss:.4f}-val_acc{" \
                  "val_acc:.4f}.hdf5 "
TRAINING_DATA_DIR = '/content/drive/My Drive/Trivago-SPA&Wellness/train'
VALIDATION_DATA_DIR = '/content/drive/My Drive/Trivago-SPA&Wellness/valid'
TEST_DATA_DIR = '/content/drive/My Drive/Trivago-SPA&Wellness/test'

CFG = {
    "data": {
        "path": {
            "train_path": TRAINING_DATA_DIR,
            "valid_path": VALIDATION_DATA_DIR,
            "test_path": TEST_DATA_DIR,
            "path": PATH
        }

    },
    "train": {
        "batch_size": BATCH_SIZE,
        "image_size": IMAGE_SIZE,
        "epochs": EPOCHS,
        "optimizer": {
            "type": "SGD",
            "lr": INIT_LR,
            "momentum": 0.9
        },
        "preprocess": {
            "rescale": 1. / 255,
            "shear_range": 0.1,
            "zoom_range": 0.1,
            "horizontal_flip": True
        },
        "metrics": ["accuracy"],
        "loss": 'categorical_crossentropy',
        "verbose": 2
    },
    "test": {
        "test_size": TEST_SIZE
    },
    "model": {
        "input": [IMAGE_SIZE, IMAGE_SIZE, 3],
        "dense1": {
            "units": 128,
            "activation": "relu",
            "kernel_initializer": "he_uniform"
        },
        "output": {
            "units": 6,
            "activation": "softmax"
        },
        "model_filename": MODEL_FILE_NAME
    }
}
