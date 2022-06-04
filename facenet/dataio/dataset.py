"""Library implementing FaceDataset.

Author
 * Mohammadamin Aliari
"""


import os
from glob import glob

import PIL
import torch
import numpy as np
from torchvision import transforms


class FaceDataset(torch.utils.data.Dataset):
    """This function implements FaceDataset for data processing and loading.

    Arguments
    ---------
    root_dir : str
        Root directory of the dataset. It should contain each class folders separatly.
    img_dim : int
        Input image dimension meaning width and height.

    Example
    -------
    >>> inp_tensor = torch.rand([10, 3, 256, 256])
    >>> dataset = FaceDataset(root_dir='data/valid', img_dim=(256, 256))
    >>> img_dim = dataset.img_dim
    >>> img_dim
    (256, 256)
    """

    labels = {'real': 0, 'fake': 1}
    label_count = len(labels)
    label_to_str = ['real', 'fake']

    def __init__(self, root_dir, img_dim):
        self.root_dir = root_dir
        self.img_dim = img_dim
        self.transform = transforms.Compose([
            transforms.Resize(self.img_dim),
            transforms.ToTensor(),
        ])

        # find images paths and their classes in all subdirs.
        self.data = []
        for path in glob(f'{self.root_dir}/*/'):
            label = os.path.basename(os.path.normpath(path))
            for img_path in glob(f'{path}/*'):
                src = os.path.basename(os.path.normpath(img_path))
                try:
                    src = src.split('_')[1].split('.')[0]
                except Exception as e:
                    e

                self.data.append([img_path, label, src])

    def __getitem__(self, idx):
        img_path, label, src = self.data[idx]
        img = PIL.Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)
        label_tensor = torch.tensor(FaceDataset.labels[label])

        return img_tensor, label_tensor, src

    def __len__(self):
        return len(self.data)

    def get_data_loader(self, batch_size, use_shuffle):
        """Returns the FaceDataset data loader.

        Arguments
        ---------
        batch_size : int
            size of each mini-batch.
        use_shuffle : bool
            shuffle mini-batches or not.

        Example
        -------
        >>> dataset = FaceDataset(root_dir='data/samples', img_dim=(256, 256))
        >>> loader = dataset.get_data_loader(batch_size=1, use_shuffle=False)
        >>> len(loader)
        0

        Returns
        ---------
        out: torch.utils.data.DataLoader
        Torch data loader containing utility to get batches, shuffling, etc.
        """

        dataset = torch.utils.data.Subset(self, np.arange(0, self.__len__()))
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=use_shuffle)

        return loader
