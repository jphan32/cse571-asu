from Data_Loaders import Data_Loaders
from Networks import Action_Conditioned_FF

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

import numpy as np
import random

def fix_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def train_model(no_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 128
    data_loaders = Data_Loaders(batch_size)
    model = Action_Conditioned_FF().to(device)

    loss_function = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    losses = []
    min_loss = model.evaluate(model, data_loaders.test_loader, loss_function)
    losses.append(min_loss)

    for epoch_i in tqdm(range(no_epochs)):
        model.train()
        for idx, sample in enumerate(data_loaders.train_loader):
            inputs = sample['input'].to(device)
            labels = sample['label'].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

        test_loss = model.evaluate(model, data_loaders.test_loader, loss_function)
        losses.append(test_loss)

    torch.save(model.state_dict(), 'saved/saved_model.pkl')


if __name__ == '__main__':
    no_epochs = 100
    fix_seed(32)
    train_model(no_epochs)
