from Data_Loaders import Data_Loaders
from Networks import Action_Conditioned_FF

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

import numpy as np
import random
import wandb
import sys

def fix_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def train_model(no_epochs, test_name="test", random_seed=32):
    fix_seed(random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 128
    learning_rate = 0.001
    early_stopping_patience = 5 
    min_test_loss = float('inf')
    epochs_no_improve = 0

    data_loaders = Data_Loaders(batch_size)
    model = Action_Conditioned_FF().to(device)

    loss_function = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    wandb.init( entity="jphan32",
                project="CSE-571",
                name=test_name,
                config={
                    "learning_rate": learning_rate,
                    "architecture": "Action_Conditioned_FF",
                    "dataset": "Custom Robot Sensor Data",
                    "epochs": no_epochs,
                    "seed": random_seed
                })

    losses = []
    min_loss = model.evaluate(model, data_loaders.test_loader, loss_function)
    losses.append(min_loss)

    for epoch_i in tqdm(range(no_epochs)):
        model.train()
        total_train_loss = 0.0
        for idx, sample in enumerate(data_loaders.train_loader):
            inputs = sample['input'].to(device)
            labels = sample['label'].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
        average_train_loss = total_train_loss / len(data_loaders.train_loader)

        average_test_loss = model.evaluate(model, data_loaders.test_loader, loss_function)

        wandb.log({"average_train_loss": average_train_loss, "average_test_loss": average_test_loss})

        if average_test_loss < min_test_loss:
            min_test_loss = average_test_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'saved/best_model.pkl')
        else:
            epochs_no_improve += 1
            if epochs_no_improve == early_stopping_patience:
                print(f'Early Stopping : {epoch_i} epoch')
                break

    torch.save(model.state_dict(), 'saved/saved_model.pkl')
    wandb.finish()


if __name__ == '__main__':
    no_epochs = 1000
    train_model(no_epochs, test_name = sys.argv[1] if len(sys.argv) > 1 else "test")
