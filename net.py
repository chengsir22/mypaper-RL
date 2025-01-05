import torch


class QNet(torch.nn.Module):
    def __init__(self, space_length, action_scale_list):
        super(QNet, self).__init__()
        self.space_length = space_length
        self.action_scale_list = action_scale_list
        self.fc1 = torch.nn.Linear(self.space_length + 1, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 64)
        self.fc4 = torch.nn.Linear(64, 1)

    def forward(self, input):
        x1 = torch.relu(self.fc1(input))
        x2 = torch.relu(self.fc2(x1))
        x3 = torch.relu(self.fc3(x2))
        return self.fc4(x3)
