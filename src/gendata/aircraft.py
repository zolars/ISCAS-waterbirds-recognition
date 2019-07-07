# -*- coding: utf-8 -*
"""This module is served as torchvision.datasets to load Aircraft dataset.
This file is modified from:
    https://github.com/vishwakftw/vision.
"""

import os
import pickle

import numpy as np
import PIL.Image
import torch


__all__ = ['GENDATA', 'GENDATAReLU']
__author__ = 'Xin Yifei'
__copyright__ = '2018 LAMDA'
__date__ = '2019-06-17'
__license__ = 'CC BY-SA 3.0'
__status__ = 'Development'
__updated__ = '2019-06-17'
__version__ = '12.0'


class GENDATA(torch.utils.data.Dataset):
    """GENDATA dataset.
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

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, extract=False):
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
            if download:
                url = None
                self._download(url)
                self._extract()
                extract = False
            elif extract:
                self._extract()
            else:
                raise RuntimeError(
                    'Dataset not found. You can use download=True to download it.')

        # Now load the picked data.
        if self._train:
            self._train_data, self._train_labels = pickle.load(open(
                os.path.join(self._root, 'processed/train.pkl'), 'rb'))
            assert (len(self._train_data) == 3334
                    and len(self._train_labels) == 3334)
        else:
            self._test_data, self._test_labels = pickle.load(open(
                os.path.join(self._root, 'processed/test.pkl'), 'rb'))
            assert (len(self._test_data) == 3333
                    and len(self._test_labels) == 3333)

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
        return (
            os.path.isfile(os.path.join(self._root, 'processed/train.pkl'))
            and os.path.isfile(os.path.join(self._root, 'processed/test.pkl')))

    def _download(self, url):
        raise NotImplementedError

    def _extract(self):
        """Prepare the data for train/test split and save onto disk."""
        image_path = os.path.join(
            self._root, 'raw/fgvc-aircraft-2013b/images/')
        # Format of images.txt: <image_id> <image_name>
        id2name = np.genfromtxt(os.path.join(
            self._root, 'raw/fgvc-aircraft-2013b/images.txt'), dtype=str)
        # Format of train_test_split.txt: <image_id> <is_training_image>
        id2train = np.genfromtxt(os.path.join(
            self._root, 'raw/fgvc-aircraft-2013b/train_test_split.txt'), dtype=int)

        train_data = []
        train_labels = []
        test_data = []
        test_labels = []
        for id_ in range(id2name.shape[0]):
            image = PIL.Image.open(os.path.join(image_path, id2name[id_, 1]))
            label = int(id2name[id_, 1][:3]) - 1  # Label starts with 0

            # Convert gray scale image to RGB image.
            if image.getbands()[0] == 'L':
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
                    open(os.path.join(self._root, 'processed/train.pkl'), 'wb'))
        pickle.dump((test_data, test_labels),
                    open(os.path.join(self._root, 'processed/test.pkl'), 'wb'))


class GENDATAReLU(torch.utils.data.Dataset):
    """GENDATA relu5-3 dataset.
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
            print('Aircraft relu5-3 features already prepared.')
        else:
            raise RuntimeError('Aircraft relu5-3 Dataset not found.'
                               'You need to prepare it in advance.')

        # Now load the picked data.
        if self._train:
            self._train_data, self._train_labels = torch.load(open(
                os.path.join(self._root, 'relu5-3/train.pth'), 'rb'))
            assert (len(self._train_data) == 3334
                    and len(self._train_labels) == 3334)
        else:
            self._test_data, self._test_labels = torch.load(open(
                os.path.join(self._root, 'relu5-3/test.pth'), 'rb'))
            assert (len(self._test_data) == 3333
                    and len(self._test_labels) == 3333)

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
        return (
            os.path.isfile(os.path.join(self._root, 'relu5-3', 'train.pth'))
            and os.path.isfile(os.path.join(self._root, 'relu5-3', 'test.pth')))