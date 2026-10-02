"""CelebA dataset loader used by the MAC/BioTask training and evaluation scripts."""

from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
import numpy
import torch


def load_celebA_data(path, split, transforms, attribute, batch_size=1, good_labels=None, shuffle=True):
    """
    Loads, saves and outputs the CelebA dataset

    https://arxiv.org/abs/1411.7766

    param path: location at which to save the images
    param split: takes values in [‘train’, ‘valid’, ‘test’, ‘all’]
    param attribute: type of attributes to download, one of ['attr', 'identity', 'bbox']
    """
    dataset = datasets.CelebA(root=path, split=split, target_type=attribute,
                              transform=transforms,
                              download=True)

    attributes_names = dataset.attr_names
    if good_labels:
        good_attr_idx = [attributes_names.index(label) for label in good_labels]
    else:
        good_attr_idx = list(range(0, len(attributes_names) - 1))

    dataset.attr = dataset.attr[:, good_attr_idx]
    attributes_names = [attributes_names[i] for i in good_attr_idx]

    class_weights = []
    for i in range(dataset.attr.size()[1]):
        numpy_weights = compute_class_weight(class_weight='balanced', classes=numpy.asarray([0, 1]),
                                             y=dataset.attr[:, i].numpy())
        torch_weights = torch.tensor(numpy_weights, dtype=torch.float)
        class_weights.append(torch_weights)

    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=50)

    return loader, attributes_names, class_weights


def load_LFW_data(path, split, batch_size=1, shuffle=False):
    """
    Loads, saves and outputs the LFW dataset

    http://vis-www.cs.umass.edu/lfw/

    param path: location at which to save the images
    param split: takes values in ['train', 'test', '10fold']
    """
    dataset = datasets.LFWPeople(root=path, split=split, image_set="funneled", download=True)

    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

    return loader
