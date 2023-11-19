import torch
import torch.utils.data as data
import torch.utils.data.dataset as dataset
from torch.utils.data import DataLoader, Subset
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import random

class Nav_Dataset(dataset.Dataset):
    def __init__(self):
        self.data = np.genfromtxt('saved/training_data.csv', delimiter=',')
        self.data = np.unique(self.data, axis=0)

        # normalize data and save scaler for inference
        self.scaler = MinMaxScaler()
        self.normalized_data = self.scaler.fit_transform(self.data) #fits and transforms
        pickle.dump(self.scaler, open("saved/scaler.pkl", "wb")) #save to normalize at inference

    def __len__(self):
        return len(self.normalized_data)

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            idx = idx.item()
        x = torch.from_numpy(self.normalized_data[idx, :6]).float()
        y = torch.from_numpy(np.array(self.normalized_data[idx, 6])).float()
        return {'input':x, 'label':y}


class Data_Loaders():
    def __init__(self, batch_size, alpha=1.0, cutmix_prob=0.5, use_cutmix=False):
        self.nav_dataset = Nav_Dataset()
        original_length = len(self.nav_dataset)

        data = [self.nav_dataset[i] for i in range(len(self.nav_dataset))]
        inputs = [d['input'] for d in data]
        labels = [d['label'] for d in data]

        indices = list(range(len(self.nav_dataset)))
        train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42, stratify=labels)

        if use_cutmix:
            augmented_dataset = []
            for i in train_indices:
                original_data = self.nav_dataset[i]['input']
                original_label = self.nav_dataset[i]['label']
                augmented_dataset.append({'input': original_data, 'label': original_label})

                random_idx = np.random.choice(train_indices)
                mix_data = self.nav_dataset[random_idx]['input']
                mix_label = self.nav_dataset[random_idx]['label']

                augmented_input, augmented_label = self.cutmix_data([original_data, mix_data], [original_label, mix_label], alpha)
                augmented_dataset.append({'input': augmented_input, 'label': augmented_label})

            self.train_loader = DataLoader(augmented_dataset, batch_size=batch_size, shuffle=True)
        else:
            train_subset = Subset(self.nav_dataset, train_indices)
            self.train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)

        test_subset = Subset(self.nav_dataset, test_indices)
        self.test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    
    def cutmix_data(self, inputs, labels, alpha=1.0):
        input1, input2 = inputs
        label1, label2 = labels

        lam = np.random.beta(alpha, alpha)
        mixed_input = lam * input1 + (1 - lam) * input2
        mixed_label = lam * label1 + (1 - lam) * label2

        return mixed_input, mixed_label


def main():
    batch_size = 16
    data_loaders = Data_Loaders(batch_size)
    # STUDENTS : note this is how the dataloaders will be iterated over, and cannot be deviated from
    for idx, sample in enumerate(data_loaders.train_loader):
        _, _ = sample['input'], sample['label']
    for idx, sample in enumerate(data_loaders.test_loader):
        _, _ = sample['input'], sample['label']

if __name__ == '__main__':
    main()
