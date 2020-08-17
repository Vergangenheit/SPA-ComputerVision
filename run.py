import os
from PIL import ImageFile
from model.model import VGGSPA
from configs.config import CFG
import argparse
import tensorflow as tf

ImageFile.LOAD_TRUNCATED_IMAGES = True
tf.get_logger().setLevel('WARNING')

# training_generator = training_generator(config.TRAINING_DATA_DIR)
# validation_generator = validation_generator(config.VALIDATION_DATA_DIR)
# ckpt_path = os.path.join(config.PATH, 'ckpt')


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model', type=tf.keras.Model, default=model.define_model(), help='what type of model and ')
    parser.add_argument('--lrs', type=str, default=None, help='defines the learning rate scheduler')
    parser.add_argument('--ckpt_path', type=str, default=os.path.join(CFG["data"]["path"]["path"], 'ckpt'),
                        help='where to save the trained models checkpoints')
    args = parser.parse_args()

    model = VGGSPA(CFG)
    model.train_generator()
    model.valid_generator()
    model.t_generator()
    model.build()
    model.train(args)


if __name__ == '__main__':
    main()
