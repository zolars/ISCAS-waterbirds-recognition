import os
import pickle
from tqdm import tqdm

import numpy as np
import torch
import PIL.Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class GENDATA(torch.utils.data.Dataset):
    """cas_bird dataset.
    Args:
        _root, str: Root directory of the dataset.
        _train, bool: Load train/test data.
        _transform, callable: A function/transform that takes in a PIL.Image
            and transforms it.
        _target_transform, callable: A function/transform that takes in the
            target and transforms it.
        _train_data, list of np.ndarray.
        _train_labels, list of int.
        _test_data, list of np.ndarray.
        _test_labels, list of int.
    """
    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False,
                 extract=False):
        """Load the dataset.
        Args
            root, str: Root directory of the dataset.
            train, bool [True]: Load train/test data.
            transform, callable [None]: A function/transform that takes in a
                PIL.Image and transforms it.
            target_transform, callable [None]: A function/transform that takes
                in the target and transforms it.
            download, bool [False]: If true, downloads the dataset from the
                internet and puts it in root directory. If dataset is already
                downloaded, it is not downloaded again.
        """
        self._root = os.path.expanduser(root)  # Replace ~ by the complete dir
        self._train = train
        self._transform = transform
        self._target_transform = target_transform

        if self._checkIntegrity():
            print('Files already downloaded and verified.')
        else:
            if extract:
                self._extract()
            else:
                raise RuntimeError(
                    'Dataset not found. You can use download=True to download it.'
                )

        # Now load the picked data.
        if self._train:
            self._train_data, self._train_labels = pickle.load(
                open(os.path.join(self._root, 'processed/train.pkl'), 'rb'))
            assert (len(self._train_data) == 16400
                    and len(self._train_labels) == 16400)
        else:
            self._test_data, self._test_labels = pickle.load(
                open(os.path.join(self._root, 'processed/test.pkl'), 'rb'))
            assert (len(self._test_data) == 11056
                    and len(self._test_labels) == 11056)

    def __getitem__(self, index):
        """
        Args:
            index, int: Index.
        Returns:
            image, PIL.Image: Image of the given index.
            target, str: target of the given index.
        """
        if self._train:
            image, target = self._train_data[index], self._train_labels[index]
        else:
            image, target = self._test_data[index], self._test_labels[index]
        # Doing this so that it is consistent with all other datasets.
        image = PIL.Image.fromarray(image)

        if self._transform is not None:
            image = self._transform(image)
        if self._target_transform is not None:
            target = self._target_transform(target)

        return image, target

    def __len__(self):
        """Length of the dataset.
        Returns:
            length, int: Length of the dataset.
        """
        if self._train:
            return len(self._train_data)
        return len(self._test_data)

    def _checkIntegrity(self):
        """Check whether we have already processed the data.
        Returns:
            flag, bool: True if we have already processed the data.
        """
        return (os.path.isfile(os.path.join(self._root, 'processed/train.pkl'))
                and os.path.isfile(
                    os.path.join(self._root, 'processed/test.pkl')))

    def _extract(self):
        """Prepare the data for train/test split and save onto disk."""
        print('Prepare the data for train/test split and save onto disk...')

        image_path = os.path.join(self._root, 'raw/04-16/images/')
        # Format of images.txt: <image_id> <image_name>
        id2name = np.genfromtxt(os.path.join(self._root,
                                             'raw/04-16/images.txt'),
                                dtype=str)
        id2class = np.genfromtxt(os.path.join(
            self._root, 'raw/04-16/image_class_labels.txt'),
                                 dtype=str)
        # Format of train_test_split.txt: <image_id> <is_training_image>
        id2train = np.genfromtxt(os.path.join(
            self._root, 'raw/04-16/train_test_split.txt'),
                                 dtype=int)

        train_data = []
        train_labels = []
        test_data = []
        test_labels = []
        for id_ in tqdm(range(id2name.shape[0])):
            image = PIL.Image.open(os.path.join(image_path, id2name[id_, 1]))
            label = int(id2class[id_, 1]) - 1  # Label starts with 0
            assert label >= 0 and label <= 164

            # Convert gray scale image to RGB image.
            # if image.getbands()[0] == 'L':
            image = image.convert('RGB')
            image_np = np.array(image)
            image.close()

            if id2train[id_, 1] == 1:
                train_data.append(image_np)
                train_labels.append(label)
            else:
                test_data.append(image_np)
                test_labels.append(label)

        pickle.dump((train_data, train_labels),
                    open(os.path.join(self._root, 'processed/train.pkl'),
                         'wb+'))
        pickle.dump((test_data, test_labels),
                    open(os.path.join(self._root, 'processed/test.pkl'),
                         'wb+'))


class GENDATAReLU(torch.utils.data.Dataset):
    """cas_bird relu5-3 dataset.
    Args:
        _root, str: Root directory of the dataset.
        _train, bool: Load train/test data.
        _train_data, list<torch.Tensor>.
        _train_labels, list<int>.
        _test_data, list<torch.Tensor>.
        _test_labels, list<int>.
    """
    def __init__(self, root, train=True):
        """Load the dataset.
        Args
            root, str: Root directory of the dataset.
            train, bool [True]: Load train/test data.
        """
        self._root = os.path.expanduser(root)  # Replace ~ by the complete dir
        self._train = train

        if self._checkIntegrity():
            print('cas_bird relu5-3 features already prepared.')
        else:
            raise RuntimeError('cas_bird relu5-3 Dataset not found.'
                               'You need to prepare it in advance.')

        # Now load the picked data.
        if self._train:
            self._train_data, self._train_labels = torch.load(
                os.path.join(self._root, 'relu5-3', 'train.pth'))
            assert (len(self._train_data) == 16400
                    and len(self._train_labels) == 16400)
        else:
            self._test_data, self._test_labels = torch.load(
                os.path.join(self._root, 'relu5-3', 'test.pth'))
            assert (len(self._test_data) == 11056
                    and len(self._test_labels) == 11056)

    def __getitem__(self, index):
        """
        Args:
            index, int: Index.
        Returns:
            feature, torch.Tensor: relu5-3 feature of the given index.
            target, int: target of the given index.
        """
        if self._train:
            return self._train_data[index], self._train_labels[index]
        return self._test_data[index], self._test_labels[index]

    def __len__(self):
        """Length of the dataset.
        Returns:
            length, int: Length of the dataset.
        """
        if self._train:
            return len(self._train_data)
        return len(self._test_data)

    def _checkIntegrity(self):
        """Check whether we have already processed the data.
        Returns:
            flag, bool: True if we have already processed the data.
        """
        return (os.path.isfile(os.path.join(
            self._root, 'relu5-3', 'train.pth')) and os.path.isfile(
                os.path.join(self._root, 'relu5-3', 'test.pth')))
