import os
from torch.utils.data import Dataset
import torch
import pickle
from sklearn.model_selection import train_test_split

class RoboDataset(Dataset):
    def __init__(self, root, path, val, split='Training', transform=None):
        self.root = root
        self.path = path
        self.val = val
        self.transform = transform
        self.split = split
        self.data = pickle.load(file=open(os.path.join(self.root, self.path), "rb"))

        if self.split == 'Training':
            self.train_data = self.data[0]
            self.train_labels = self.data[1]

        elif self.split == 'Train and validation':
            features = self.data[0]
            labels = self.data[1]
            self.train_x, self.val_x, self.train_y, self.val_y = train_test_split(
                features, labels, test_size=0.1, random_state=42)
        else:
            self.test_data = self.data[0]
            self.test_labels = self.data[1]

    def __getitem__(self, index):
        if self.split == 'Training':
            features, target = self.train_data[index], self.train_labels[index]

        else:
            features, target = self.test_data[index], self.test_labels[index]

        features = features.reshape(-1,1)
        if self.transform is not None:
            features = self.transform(features)
            target = torch.from_numpy(target)
        return features, target

    def __len__(self):
        if self.split == 'Training':
            return len(self.train_data)
        elif self.split == 'Train and validation':
            if self.val == False:
                return len(self.train_x)
            else:
                return len(self.val_x)
        else:
            return len(self.test_data)

def Get_Loader(root, path, split, transform, val, batch_size, shuffle, num_workers):

    Robo = RoboDataset(root=root, path=path, val=val,
                       transform=transform, split=split)

    data_loader = torch.utils.data.DataLoader(dataset=Robo,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers)

    return data_loader
