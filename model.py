# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#from torch.utils.data import DataLoader


## This tutorial is not so much about novel architectural designs so we keep things 
## simple and make use of a typical CNN that is adequate for the MNIST image classification task.

# class Net(nn.Module):
#     def __init__(self, num_classes: int) -> None:
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 4 * 4, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, num_classes)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 4 * 4)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=5)  # 1D Convolution
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=6, out_channels=16, kernel_size=5)  # 1D Convolution
        self.fc1 = nn.Linear(16 * 4, 120)  # Adjust the input size based on the output from conv layers
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # Add a channel dimension: (batch_size, 1, 30)
        x = self.pool(F.relu(self.conv1(x)))  # Output shape: (batch_size, 6, 26)
        x = self.pool(F.relu(self.conv2(x)))  # Output shape: (batch_size, 16, 9)
        x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layers: (batch_size, 16 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
