#Contributors: Dr. Anagnostopoulos
#              Jake Horio
#              Mason Mac
#Last Modified: 10/31/24
#Description: Classes for the different DNN architectures

import torch
from torch import nn
import torch.nn.functional as F

#MLP architecture with 4 hidden layers
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        #input fc layer (28 * 28 = 784 input neurons)
        self.fc1 = nn.Linear(784, 512)
        #hidden fc layer 1
        self.fc2 = nn.Linear(512, 64)
        #hidden fc layer 2
        self.fc3 = nn.Linear(64, 32)
        #hidden fc layer 3
        self.fc4 = nn.Linear(32, 64)
        #hidden fc layer 4
        self.fc5 = nn.Linear(64, 512)
        #output fc layer
        self.fc6 = nn.Linear(512, 10)
        self.relu = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.log_softmax(self.fc6(x))
        return x

#CNN architecture with 3 convolutional layers
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()      
        #convolutional layer 1 
        #input channels = 1, output channels = 6, kernel = 5x5
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)  # output: 28x28 -> 28x28
        #pooling layer 1 
        #2x2 max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  
        #output: 28x28 -> 14x14        
        #convolutional layer 2 
        #input channels = 6, output channels = 16, kernel = 5x5
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        #output: 14x14 -> 10x10       
        #convolutional layer 3
        #input channels = 16, output channels = 32, kernel = 3x3
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        #output: 5x5 -> 3x3 (after pooling)        
        #fully connected layers
        #input features = 32 * 3 * 3 from conv3 + pool
        self.fc1 = nn.Linear(32 * 3 * 3, 120)
        self.fc2 = nn.Linear(120, 84)
        #output of 10 neurons for the 0-9 MNIST digit classes
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        #flatten the tensor to expect a 28 * 28 input size
        x = x.view(-1, 1, 28, 28)
        #convolution layer 1 followed by ReLU and pooling layer 1
        x = self.pool(F.relu(self.conv1(x)))
        #output: 28x28 -> 14x14 (after pooling)
        #convolution layer 2 followed by ReLU and pooling layer 2
        x = self.pool(F.relu(self.conv2(x)))  
        #output: 10x10 -> 5x5 (after pooling)      
        #convolution layer 3 followed by ReLU (no pooling)
        x = F.relu(self.conv3(x))  
        #output: 5x5 -> 3x3       
        #flatten the tensor for fully connected layers
        x = x.view(-1, 32 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))       
        x = F.log_softmax(x, dim=1)
        return x


