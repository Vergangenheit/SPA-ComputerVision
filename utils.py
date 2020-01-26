import os
import shutil

from PIL import Image


def check_image_with_pil(path):
    for cls in os.listdir(path):
        print(cls, '\n')
        images = os.listdir(os.path.join(path, cls))
        for image in images:
            print(image)
            try:
                Image.open(os.path.join(path, cls, image))
            except IOError:
                print(image, False)
                os.remove(os.path.join(path, cls, image))
                continue
            pass

        print('\n')


class download_and_sort_images(object):

    def __init__(self, split_ratio, file):
        self.split_ratio = split_ratio
        self.file = file
        self.counts = []
        self.splits = []
        self.classes = []
        self.cls_splits = {}
        df = pd.read_csv(self.file)
        is_image = df['image_url'].str.contains('jpg', regex=False)
        self.data = df[is_image]

    def calculate_counts(self):

        self.classes = self.data['class'].value_counts().index
        self.counts = self.data['class'].value_counts().tolist()
        self.splits = [int(x * self.split_ratio) for x in self.counts]
        self.cls_splits = dict(zip(self.classes, self.splits))
        return self.cls_splits

    def download_and_store(self, path, image_folder):
        for c, value in self.cls_splits.items():
            i = 1
            print(c, value)
            os.makedirs(os.path.join(path, 'train', c))
            os.makedirs(os.path.join(path, 'valid', c))
            # create or check if directories already exist
            print('\n')
            for image in [x for x in os.listdir(os.path.join(path, image_folder)) if x.split('_')[0] == c]:
                print(image)

                if len(os.listdir(os.path.join(path, 'train', c))) <= value:
                    if not os.path.exists(os.path.join(path, 'train', c, image)):
                        shutil.move(os.path.join(path, image_folder, image),
                                    os.path.join(path, 'train', c), copy_function=shutil.copytree)
                    else:
                        continue

                else:
                    if not os.path.exists(os.path.join(path, 'valid', c, image)):
                        shutil.move(os.path.join(path, image_folder, image),
                                    os.path.join(path, 'valid', c), copy_function=shutil.copytree)
                    else:
                        continue

            print('Moved all train and validation images for class {}'.format(c))
            print('\n')


def move_to_test(valid_path, test_path):
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    for folder in os.listdir(valid_path):
        length = len(os.listdir(os.path.join(valid_path, folder)))
        print(int(length * 0.5), '\n')
        count = 0
        for image in os.listdir(os.path.join(valid_path, folder))[:int(length * 0.5)]:
            print(os.path.join(valid_path, folder, image))
            if not os.path.exists(os.path.join(test_path, image)):
                shutil.move(os.path.join(valid_path, folder, image),
                            test_path, copy_function=shutil.copytree)
            else:
                continue
