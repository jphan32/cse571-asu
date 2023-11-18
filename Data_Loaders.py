import torch
import torch.utils.data as data
import torch.utils.data.dataset as dataset
from torch.utils.data import DataLoader, Subset
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

class Nav_Dataset(dataset.Dataset):
    def __init__(self):
        self.data = np.genfromtxt('saved/training_data.csv', delimiter=',')

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
    def __init__(self, batch_size):
        self.nav_dataset = Nav_Dataset()

        data = [self.nav_dataset[i] for i in range(len(self.nav_dataset))]
        inputs = [d['input'] for d in data]
        labels = [d['label'] for d in data]

        indices = list(range(len(self.nav_dataset)))
        train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42, stratify=labels)

        train_subset = Subset(self.nav_dataset, train_indices)
        test_subset = Subset(self.nav_dataset, test_indices)

        self.train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)


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
