from torch.utils.data import Subset

import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


import numpy as np

"""
label mapping:

PLAID_2017:
{0: 'CFL', 1: 'Hairdryer', 2: 'Microwave', 3: 'AC', 4: 'Laptop', 5: 'Vacuum',
 6: 'ILB', 7: 'WM', 8: 'Fan', 9: 'Heater', 10: 'Fridge'}

COOLL:
{0: 'Drill', 1: 'Fan', 2: 'Grinder', 3: 'Hair', 4: 'Hedge', 
5: 'Lamp', 6: 'Paint', 7: 'Planer', 8: 'Router', 9: 'Sander', 10: 'Saw', 11: 'Vacuum'}

"""

def load_data(dataset_name):

    X = np.load(f'data/{dataset_name}/hsv_vi.npy')
    y = np.load(f'data/{dataset_name}/labels.npy')

    return X, y

class Dataset(torch.utils.data.Dataset):

    def __init__(self, feature, label, width=50):
        self.feature = feature
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        feature = self.feature[index]
        label = self.label[index]

        return feature, label


def get_loaders(input_tra, input_val, label_tra, label_val,
                batch_size=32):

    tra_data = Dataset(input_tra, label_tra)
    val_data = Dataset(input_val, label_val)

    tra_loader = torch.utils.data.DataLoader(tra_data, batch_size, shuffle=True, drop_last=False)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size, shuffle=False, drop_last=False)

    loaders = {'train': tra_loader, 'test': val_loader}

    return loaders


def split_open_set(data, labels, unknown_class: list[int], test_size=0.3, random_state=42):
    # Convert known_classes to a set for faster membership testing
    unknown_classes_set = unknown_class

    # Identify known and unknown indices based on labels
    known_indices = [i for i, label in enumerate(labels) if label not in unknown_classes_set]
    unknown_indices = [i for i, label in enumerate(labels) if label in unknown_classes_set]

    # Split known data
    known_data = data[known_indices]
    known_labels = labels[known_indices]

    # Split unknown data
    unknown_data = data[unknown_indices]
    unknown_labels = labels[unknown_indices]

    # Split the known data into training and testing sets
    train_data, test_known_data, train_labels, test_known_labels = train_test_split(
        known_data, known_labels, test_size=test_size, random_state=random_state, stratify=known_labels
    )

    le = LabelEncoder()
    le.fit(train_labels)

    train_labels = le.transform(train_labels)
    test_known_labels = le.transform(test_known_labels)
    unknown_labels[:] = len(np.unique(train_labels))

    # Combine known and unknown data for the test set
    test_data = np.concatenate((test_known_data, unknown_data), axis=0)
    test_labels = np.concatenate((test_known_labels, unknown_labels), axis=0)

    return train_data, train_labels, test_data, test_labels



