import chainer
import cv2
import os
import pandas as pd

class ImageDataset(chainer.dataset.DatasetMixin):
    def __init__(self, root, path, exist_label=False):
        self.root = root
        self.exist_label = exist_label
        self.paths = pd.read_csv(os.path.join(root, path), header=None)[0].tolist()

    def __len__(self):
        return len(self.paths)

    def get_example(self, i):  # これの後にやること: /255, float32, transpose(現在HWC), trainのときはflip
        path = self.paths[i].split(' ')[0]
        image = cv2.imread(os.path.join(self.root, path))
        assert image is not None, 'Image is None. {}'.format(path)
        return image
