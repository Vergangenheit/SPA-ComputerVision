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