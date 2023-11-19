import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class Action_Conditioned_FF(nn.Module):
    def __init__(self):
        super(Action_Conditioned_FF, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.sensor_fc1 = nn.Linear(5, 64)
        self.sensor_bn1 = nn.BatchNorm1d(64)
        self.sensor_dropout1 = nn.Dropout(0.5)

        self.sensor_fc2 = nn.Linear(64, 128)
        self.sensor_bn2 = nn.BatchNorm1d(128)
        self.sensor_dropout2 = nn.Dropout(0.5)

        self.action_fc1 = nn.Linear(1, 32)

        self.combined_fc1 = nn.Linear(128 + 32, 64)
        self.combined_bn1 = nn.BatchNorm1d(64)
        self.combined_dropout1 = nn.Dropout(0.5)

        self.fc_out = nn.Linear(64, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input):
        if input.dim() == 1:
            input = input.unsqueeze(0)
        
        sensor_data = input[:, 0:5]
        action_data = input[:, 5].unsqueeze(1)

        sensor_data = F.relu(self.sensor_bn1(self.sensor_fc1(sensor_data)))
        sensor_data = self.sensor_dropout1(sensor_data)
        sensor_data = F.relu(self.sensor_bn2(self.sensor_fc2(sensor_data)))
        sensor_data = self.sensor_dropout2(sensor_data)

        action_data = F.relu(self.action_fc1(action_data))

        combined_data = torch.cat((sensor_data, action_data), dim=1)
        combined_data = F.relu(self.combined_bn1(self.combined_fc1(combined_data)))
        combined_data = self.combined_dropout1(combined_data)

        output = self.fc_out(combined_data)
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
