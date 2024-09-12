import torch
import torch.nn as nn
import torch.nn.functional as F


##Input is 30
# class Net(nn.Module):
#     def __init__(self, num_classes: int) -> None:
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=5)  # 1D Convolution
#         self.pool = nn.MaxPool1d(kernel_size=2)
#         self.conv2 = nn.Conv1d(in_channels=6, out_channels=16, kernel_size=5)  # 1D Convolution
#         self.fc1 = nn.Linear(16 * 4, 120)  # Adjust the input size based on the output from conv layers
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, num_classes)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = x.unsqueeze(1)  # Add a channel dimension: (batch_size, 1, 30)
#         x = self.pool(F.relu(self.conv1(x)))  # Output shape: (batch_size, 6, 26)
#         x = self.pool(F.relu(self.conv2(x)))  # Output shape: (batch_size, 16, 9)
#         x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layers: (batch_size, 16 * 4)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
    

#input is 40

class Net(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=5)  # 1D Convolution
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=6, out_channels=16, kernel_size=5)  # 1D Convolution
        
        # Recalculate the input size for the fully connected layer based on new input size
        # New input is 40, conv1 reduces it to (40 - 5 + 1) = 36, pooling reduces it to 18
        # conv2 reduces it to (18 - 5 + 1) = 14, pooling reduces it to 7
        self.fc1 = nn.Linear(16 * 7, 120)  # Adjust the input size based on the output from conv layers
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # Add a channel dimension: (batch_size, 1, 40)
        x = self.pool(F.relu(self.conv1(x)))  # Output shape: (batch_size, 6, 36), after pooling: (batch_size, 6, 18)
        x = self.pool(F.relu(self.conv2(x)))  # Output shape: (batch_size, 16, 14), after pooling: (batch_size, 16, 7)
        x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layers: (batch_size, 16 * 7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

