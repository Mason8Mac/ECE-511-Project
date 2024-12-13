#Contributors: Jake Horio
#              Mason Mac
#Last Modified: 12/12/24
#Description: Classes for the different CNN-based DNN architectures with 3x3 kernels

import torch
from torch import nn
import torch.nn.functional as F

#single convolutional layer 3x3 kernel LeNet CNN with CIFAR-10 dataset
class CNN_Conv_1_CIFAR10(nn.Module):
    def __init__(self):
        super(CNN_Conv_1_CIFAR10, self).__init__()
        #convolutional layer 1 with 3 RGB input channels and 6 output feature maps
        self.conv1 = nn.Conv2d(3, 6, 3, padding = 1)        
        #2x2 max pooling kernel
        self.pool = nn.MaxPool2d(2, 2)
        #10 outputs to correpsond with CIFAR-10 images
        self.fc1 = nn.Linear(6 * 16 * 16, 10)
        
    def forward(self, x):
        #convolutional layer with ReLU and max pooling
        x = self.pool(F.relu(self.conv1(x)))
        #flatten output for fully connected layer
        x = x.view(-1, 6 * 16 * 16)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)

#LeNet 3-convolutional-layer 3x3 kernel CNN with CIFAR-10 dataset
class CNN_LeNet_CIFAR10(nn.Module):
    def __init__(self):
        super(CNN_LeNet_CIFAR10, self).__init__()
        #convolutional layer 1 with 3 RGB input channels and 6 output feature maps
        self.conv1 = nn.Conv2d(3, 6, 3, padding=1)
        #convolutional layer 2 with 6 input channels and 16 output feature maps
        self.conv2 = nn.Conv2d(6, 16, 3, padding=1)
        #convolutional layer 2 with 16 input channels and 32 output feature maps
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        #2x2 max pooling kernel
        self.pool = nn.MaxPool2d(2, 2)
        #fully connected layer 1 with 120 outputs
        self.fc1 = nn.Linear(32 * 4 * 4, 120)
        #fully connected layer 2 with 84 outputs
        self.fc2 = nn.Linear(120, 84)
        #fully connected layer 3 with 10 outputs for CIFAR-10 image classifications
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        #convolutional layers with ReLU and max pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        #flatten output for fully connected layers
        x = x.view(-1, 32 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(x, dim=1)
        return x

#5-convolutional-layer 3x3 kernel CNN with CIFAR-10 dataset
class CNN_Conv_5_CIFAR10(nn.Module):
    def __init__(self):
        super(CNN_Conv_5_CIFAR10, self).__init__()
        #convolutional layer 1 with 3 input channels and 6 output feature maps
        self.conv1 = nn.Conv2d(3, 6, 3, padding=1)
        #convolutional layer 2 with 6 input channels and 16 output feature maps
        self.conv2 = nn.Conv2d(6, 16, 3, padding=1)
        #convolutional layer 3 with 16 input channels and 32 output feature maps
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        #convolutional layer 4 with 32 input channels and 64 output feature maps
        self.conv4 = nn.Conv2d(32, 64, 3, padding=1)
        #convolutional layer 5 with 64 input channels and 128 output feature maps
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        #2x2 max pooling kernel
        self.pool = nn.MaxPool2d(2, 2)
        #fully connected layer 1 with 120 outputs
        self.fc1 = nn.Linear(128 * 1 * 1, 120)
        #fully connected layer 2 with 84 outputs
        self.fc2 = nn.Linear(120, 84)
        #fully connected layer 3 with 10 outputs for CIFAR-10 image classifications
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        #convolutional layers with ReLU and max pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        #flatten output for fully connected layers
        x = x.view(-1, 128 * 1 * 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(x, dim=1)
        return x
    
#7-convolutional-layer 3x3 kernel CNN with CIFAR-10 dataset
class CNN_Conv_7_CIFAR10(nn.Module):
    def __init__(self):
        super(CNN_Conv_7_CIFAR10, self).__init__()
        #convolutional layer 1 with 3 input channels and 6 output feature maps
        self.conv1 = nn.Conv2d(3, 6, 3, padding=1)
        #convolutional layer 2 with 6 input channels and 16 output feature maps
        self.conv2 = nn.Conv2d(6, 16, 3, padding=1)
        #convolutional layer 3 with 16 input channels and 32 output feature maps
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        #convolutional layer 4 with 32 input channels and 64 output feature maps
        self.conv4 = nn.Conv2d(32, 64, 3, padding=1)
        #convolutional layer 5 with 64 input channels and 128 output feature maps
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        #convolutional layer 6 with 128 input channels and 256 output feature maps
        self.conv6 = nn.Conv2d(128, 256, 3, padding=1)
        #convolutional layer 7 with 256 input channels and 512 output feature maps
        self.conv7 = nn.Conv2d(256, 512, 3, padding=1)
        #2x2 max pooling kernel
        self.pool = nn.MaxPool2d(2, 2)
        #fully connected layer 1 with 120 outputs
        self.fc1 = nn.Linear(512 * 2 * 2, 120)
        #fully connected layer 2 with 84 outputs
        self.fc2 = nn.Linear(120, 84)
        #fully connected layer 3 with 10 outputs for CIFAR-10 image classifications
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        #convolutional layers with ReLU and alternating max pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.pool(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = self.pool(F.relu(self.conv5(x)))
        x = F.relu(self.conv6(x))
        x = self.pool(F.relu(self.conv7(x)))
        #flatten output for fully connected layers
        x = x.view(-1, 512 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(x, dim=1)
        return x

#4-block 3x3 kernel VGG CNN with CIFAR-10 dataset
class VGG_CIFAR10(nn.Module):
    def __init__(self):
        super(VGG_CIFAR10, self).__init__()
        self.conv_layers = nn.Sequential(
            #block 1 with 3 input channels and 64 output feature maps
            #extracts low-level features such as edges
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            #block 2 with 64 input channels and 128 output feature maps
            #extracts slightly more complex features
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            #block 3 with 128 input channels and 256 output feature maps
            #extracts heavier complex features
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            #block 4 with 256 input channels and 512 output feature maps
            #extracts higher level abstract features
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
        )

        self.fc_layers = nn.Sequential(
            #flatten output for 32x32 CIFAR-10 images
            nn.Linear(512 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            #output of 10 channels for CIFAR-10
            nn.Linear(4096, 10),
        )
    
    def forward(self, x):
        #feed forward through the convolutional layers
        x = self.conv_layers(x)
        #flatten output for fully connected layers
        x = x.view(x.size(0), -1)
        #feed forward through the fully connected layers
        x = self.fc_layers(x)
        return F.log_softmax(x, dim=1)
