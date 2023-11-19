import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class Action_Conditioned_FF(nn.Module):
    def __init__(self):
        super(Action_Conditioned_FF, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.fc1 = nn.Linear(6, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(64, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.5)

        self.fc_out = nn.Linear(64, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input):
        if input.dim() == 1:
            input = input.unsqueeze(0)
        
        x = F.relu(self.bn1(self.fc1(input)))
        #x = self.dropout1(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)

        output = self.fc_out(x)
        return output.squeeze()

    def evaluate(self, model, test_loader, loss_function):
        model.eval()

        total_loss = 0.0
        with torch.no_grad():
            for data in test_loader:
                inputs = data['input'].to(self.device)
                labels = data['label'].to(self.device)

                outputs = model(inputs)

                loss = loss_function(outputs, labels)
                total_loss += loss.item()

        average_loss = total_loss / len(test_loader)
        return average_loss

def main():
    model = Action_Conditioned_FF()

if __name__ == '__main__':
    main()
