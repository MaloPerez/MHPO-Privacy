"""FairFace/CelebA/triplet dataset loaders used throughout the training and HPO
pipeline."""

import os
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import datasets


class TripletFaceDataset(Dataset):

    def __init__(self, root_dir, csv_name, num_triplets, transform=None):

        self.root_dir = root_dir
        self.df = pd.read_csv(csv_name)
        self.num_triplets = num_triplets
        self.transform = transform
        self.training_triplets = self.generate_triplets(self.df, self.num_triplets, self.root_dir)

    @staticmethod
    def generate_triplets(df, num_triplets, root_dir):

        def make_dictionary_for_face_class(df):

            """
              - face_classes = {'class0': [class0_id0, ...], 'class1': [class1_id0, ...], ...}
            """
            face_classes = dict()
            for idx, label in enumerate(df['class']):
                if label not in face_classes:
                    face_classes[label] = []
                face_classes[label].append((df.iloc[idx]['id'], df.iloc[idx]['ext']))
            return face_classes

        triplets = []
        classes = df['class'].unique()

        if os.path.isfile(os.path.join(root_dir, 'class_dictionary.pkl')):
            with open(os.path.join(root_dir, 'class_dictionary.pkl'), 'rb') as f:
                face_classes = pickle.load(f)
        else:
            face_classes = make_dictionary_for_face_class(df)
            with open(os.path.join(root_dir, 'class_dictionary.pkl'), 'wb') as f:
                pickle.dump(face_classes, f)

        for _ in range(num_triplets):

            '''
              - randomly choose anchor, positive and negative images for triplet loss
              - anchor and positive images in pos_class
              - negative image in neg_class
              - at least, two images needed for anchor and positive images in pos_class
              - negative image should have different class as anchor and positive images by definition
            '''

            pos_class = np.random.choice(classes)
            neg_class = np.random.choice(classes)
            while len(face_classes[pos_class]) < 2:
                pos_class = np.random.choice(classes)
            while pos_class == neg_class:
                neg_class = np.random.choice(classes)

            pos_name = df.loc[df['class'] == pos_class, 'name'].values[0]
            neg_name = df.loc[df['class'] == neg_class, 'name'].values[0]

            if len(face_classes[pos_class]) == 2:
                ianc, ipos = np.random.choice(2, size=2, replace=False)
            else:
                ianc = np.random.randint(0, len(face_classes[pos_class]))
                ipos = np.random.randint(0, len(face_classes[pos_class]))
                while ianc == ipos:
                    ipos = np.random.randint(0, len(face_classes[pos_class]))
            ineg = np.random.randint(0, len(face_classes[neg_class]))

            anc_id = face_classes[pos_class][ianc][0]
            anc_ext = face_classes[pos_class][ianc][1]
            pos_id = face_classes[pos_class][ipos][0]
            pos_ext = face_classes[pos_class][ipos][1]
            neg_id = face_classes[neg_class][ineg][0]
            neg_ext = face_classes[neg_class][ineg][1]

            triplets.append(
                [anc_id, pos_id, neg_id, pos_class, neg_class, pos_name, neg_name, anc_ext, pos_ext, neg_ext])

        return triplets

    def __getitem__(self, idx):

        anc_id, pos_id, neg_id, pos_class, neg_class, pos_name, neg_name, anc_ext, pos_ext, neg_ext = \
            self.training_triplets[idx]

        anc_img = os.path.join(self.root_dir, '{:07d}'.format(pos_name), '{:03d}'.format(anc_id) + f'.{anc_ext}')
        pos_img = os.path.join(self.root_dir, '{:07d}'.format(pos_name), '{:03d}'.format(pos_id) + f'.{pos_ext}')
        neg_img = os.path.join(self.root_dir, '{:07d}'.format(neg_name), '{:03d}'.format(neg_id) + f'.{neg_ext}')

        to_pil_image = transforms.ToPILImage()

        anc_img = to_pil_image(io.imread(anc_img))
        pos_img = to_pil_image(io.imread(pos_img))
        neg_img = to_pil_image(io.imread(neg_img))

        pos_class = torch.from_numpy(np.array([pos_class]).astype('long'))
        neg_class = torch.from_numpy(np.array([neg_class]).astype('long'))

        sample = {'anc_img': anc_img, 'pos_img': pos_img, 'neg_img': neg_img, 'pos_class': pos_class,
                  'neg_class': neg_class}

        if self.transform:
            sample['anc_img'] = self.transform(sample['anc_img'])
            sample['pos_img'] = self.transform(sample['pos_img'])
            sample['neg_img'] = self.transform(sample['neg_img'])

        return sample

    def __len__(self):
        return len(self.training_triplets)


class FairFaceDataset(Dataset):

    def get_class_weights(self, df):
        class_weights = []
        attributes = ['gender', 'race', 'age']
        NUM_CLASS_DICT = {'age': 9,
                          'race': 7,
                          'gender': 2}
        for attr in attributes:
            numpy_weights = compute_class_weight(class_weight='balanced',
                                                 classes=np.asarray(range(NUM_CLASS_DICT[attr])),
                                                 y=df[attr].to_numpy())
            torch_weights = torch.tensor(numpy_weights, dtype=torch.float)
            class_weights.append(torch_weights)

        return class_weights

    def __init__(self, root_dir, csv_name, transform=None):
        self.root_dir = root_dir
        self.df = pd.read_csv(csv_name).sample(frac=1).reset_index(drop=True)
        self.transform = transform
        self.label_dictionnary = {
            'race': {
                'White': 0,
                'Black': 1,
                'Latino_Hispanic': 2,
                'East Asian': 3,
                'Southeast Asian': 4,
                'Indian': 5,
                'Middle Eastern': 6
            },
            'gender': {
                'Male': 0,
                'Female': 1,
            },
            'age': {
                '0-2': 0,
                '3-9': 1,
                '10-19': 2,
                '20-29': 3,
                '30-39': 4,
                '40-49': 5,
                '50-59': 6,
                '60-69': 7,
                'more than 70': 8
            },
            'white': {
                'White': 0,
                'Black': 1,
                'Latino_Hispanic': 1,
                'East Asian': 1,
                'Southeast Asian': 1,
                'Indian': 1,
                'Middle Eastern': 1
            },
            'age2': {
                '0-2': 0,
                '3-9': 0,
                '10-19': 1,
                '20-29': 1,
                '30-39': 1,
                '40-49': 2,
                '50-59': 2,
                '60-69': 2,
                'more than 70': 2
            },
        }

        self.df['white'] = self.df['race'].map(self.label_dictionnary["white"])
        self.df['age2'] = self.df['age'].map(self.label_dictionnary["age2"])
        self.df['race'] = self.df['race'].map(self.label_dictionnary["race"])
        self.df['gender'] = self.df['gender'].map(self.label_dictionnary["gender"])
        self.df['age'] = self.df['age'].map(self.label_dictionnary["age"])

        self.class_weights = self.get_class_weights(self.df)

    def __getitem__(self, idx):

        img_info = self.df.iloc[idx].values
        img_path = img_info[0]
        img_age = np.array([img_info[1]]).astype('long')
        img_gender = np.array([img_info[2]]).astype('long')
        img_race = np.array([img_info[3]]).astype('long')
        img_white = np.array([img_info[5]]).astype('long')
        img_age2 = np.array([img_info[6]]).astype('long')

        img_path = os.path.join(self.root_dir, str(img_path))
        img = io.imread(img_path)

        sample = {'img': img, 'age': img_age, 'gender': img_gender, 'race': img_race, 'white': img_white,
                  'age2': img_age2}

        if self.transform:
            sample['img'] = self.transform(sample['img'])

        return sample

    def __len__(self):
        return len(self.df)


def load_celebA_data(path, split, attribute, transform):
    """
    Loads, saves and outputs the CelebA dataset

    https://arxiv.org/abs/1411.7766

    param path: location at which to save the images
    param split: takes values in [‘train’, ‘valid’, ‘test’, ‘all’]
    param attribute: type of attributes to download, one of ['attr', 'identity', 'bbox']
    """
    dataset = datasets.CelebA(root=path, split=split, target_type=attribute,
                              transform=transforms.Compose([transforms.ToTensor(), transform]),
                              download=True)

    class_weights = []
    for i in range(dataset.attr.size()[1]):
        numpy_weights = compute_class_weight(class_weight='balanced', classes=np.asarray([0, 1]),
                                             y=dataset.attr[:, i].numpy())
        torch_weights = torch.tensor(numpy_weights, dtype=torch.float)
        class_weights.append(torch_weights)

    attributes = dataset.attr_names

    return dataset, attributes, class_weights
