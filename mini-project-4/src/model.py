import torch
import torch.nn as nn

# Baseline model 784 → 128 → 10
class FashionClassifier(nn.Module):
    def __init__(self):
        super(FashionClassifier, self).__init__()
        
        # Fully connected layers
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        # Flatten the image: (batch, 1, 28, 28) → (batch, 784)
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x

# Deeper fully connected neural network
# Architecture: 784 → 256 → 128 → 10
class DeepFashionClassifier(nn.Module):
    def __init__(self):
        super(DeepFashionClassifier, self).__init__()
        
        self.fc1 = nn.Linear(784, 256)
        self.relu1 = nn.ReLU()
        
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        
        self.fc3 = nn.Linear(128, 10)
        
    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        
        return x

# Deeper fully connected neural network with dropout
# Architecture: 784 → 256 → Dropout → 128 → Dropout → 10
class DropoutFashionClassifier(nn.Module):
    def __init__(self):
        super(DropoutFashionClassifier, self).__init__()
        
        self.fc1 = nn.Linear(784, 256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(128, 10)
        
    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x
