import torch.nn as nn
import torch.nn.functional as F

class CNNBinaryClassifier(nn.Module):
    def __init__(self):
        super(CNNBinaryClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 1 * 1, 256)  # Adjusted input size
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        # Input size is 5x6
        x = x.view(-1, 1, 5, 6)  # Reshape to 2D (batch_size, channels, height, width)
        x = F.relu(self.conv1(x))  # Output: (batch_size, 32, 5, 6)
        x = F.max_pool2d(x, (2, 3))  # Output: (batch_size, 32, 2, 2)
        x = F.relu(self.conv2(x))  # Output: (batch_size, 64, 2, 2)
        x = F.max_pool2d(x, 2)  # Output: (batch_size, 64, 1, 1)
        x = F.relu(self.conv3(x))  # Output: (batch_size, 128, 1, 1)
        x = x.view(-1, 128 * 1 * 1)  # Flatten for fully connected layers
        x = F.relu(self.fc1(x))  # Output: (batch_size, 256)
        x = self.fc2(x)  # Output: (batch_size, 2)
        return x
