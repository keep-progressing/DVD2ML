from .base import *
import os

class SOP(BaseDataset):

    def __init__(self, root, mode, transform=None):

        self.root = os.path.join(root, 'Stanford_Online_Products')

        self.mode = mode
        self.transform = transform

        self.ys = []
        self.I = []
        self.im_paths = []

        if mode == 'train':
            metadata_path = os.path.join(
                self.root,
                'Ebay_train.txt'
            )

        elif mode == 'eval':
            metadata_path = os.path.join(
                self.root,
                'Ebay_test.txt'
            )

        else:
            raise ValueError(mode)

        BaseDataset.__init__(
            self,
            self.root,
            self.mode,
            self.transform
        )

        with open(metadata_path, 'r') as f:

            lines = f.readlines()[1:]  # skip header

            for line in lines:

                image_id, class_id, super_id, path = \
                    line.strip().split()

                self.ys.append(int(class_id) - 1)

                self.I.append(int(image_id) - 1)

                self.im_paths.append(
                    os.path.join(
                        self.root,
                        'images',
                        path
                    )
                )

        self.classes = sorted(list(set(self.ys)))