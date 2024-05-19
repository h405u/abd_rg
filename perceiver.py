import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Define the CNN Model
class CNN_MNIST(nn.Module):
    def __init__(self):
        super(CNN_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(7*7*64, 256)
        self.fc2 = nn.Linear(256, 10)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # Prepare input: expect input of shape [batchsize, input_length, 28, 28]
        # Reshape to [batchsize * input_length, 1, 28, 28]
        batchsize, input_length, height, width = x.shape
        x = x.view(-1, 1, height, width)
        
        # Forward pass
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 7*7*64)  # Flatten the outputs for each image
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        # Change the shape to [batchsize, input_length, 10]
        x = x.view(batchsize, input_length, -1)
        return x


class CNN_CIFAR10(nn.Module):
    def __init__(self):
        super(CNN_CIFAR10, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3_bn = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)
        
        self.fc1 = nn.Linear(128 * 4 * 4, 512)  # Flatten size is 128x4x4 after 3 pooling layers
        self.fc1_bn = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # Prepare input: expect input of shape [batchsize, input_length, 32, 32]
        # Reshape to [batchsize * input_length, 3, 32, 32]
        batchsize, input_length, _, height, width = x.shape
        x = x.view(-1, 3, height, width)
        
        # Forward pass
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = self.pool(x)
        
        x = self.dropout1(x)
        x = x.view(-1, 128 * 4 * 4)  # Flatten the outputs for each image
        
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = self.dropout2(x)
        x = self.fc2(x)
        
        # Change the shape to [batchsize, input_length, 10]
        x = x.view(batchsize, input_length, -1)
        return x


def train_model(model, train_loader, device):
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    model.train()
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        # Reshape outputs and labels for loss computation
        outputs = outputs.view(-1, 10)
        labels = labels.view(-1)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, -1)
            total += labels.size(0) * labels.size(1)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    return accuracy
