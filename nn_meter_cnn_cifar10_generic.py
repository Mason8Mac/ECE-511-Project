#Contributors: Jake Horio
#              Mason Mac
#Last Modified: 12/12/24
#Description: Main code module that uses the nn-Meter module tool to measure the 
#             feed-forward latency on CNN-based DNNs running on targeted new-edge hardware

import sys
import inspect
import time
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms

#classes of CNN architectures corresponding to different kernel sizes
import kernel_3x3_model_classes
import kernel_5x5_model_classes
import kernel_7x7_model_classes

from kernel_3x3_model_classes import (CNN_Conv_1_CIFAR10,
                                      CNN_LeNet_CIFAR10,
                                      CNN_Conv_5_CIFAR10,
                                      CNN_Conv_7_CIFAR10,
                                      VGG_CIFAR10)

from kernel_5x5_model_classes import (CNN_Conv_1_CIFAR10,
                                      CNN_LeNet_CIFAR10,
                                      CNN_Conv_5_CIFAR10,
                                      CNN_Conv_7_CIFAR10,
                                      VGG_CIFAR10)

from kernel_7x7_model_classes import (CNN_Conv_1_CIFAR10,
                                      CNN_LeNet_CIFAR10,
                                      CNN_Conv_5_CIFAR10,
                                      CNN_Conv_7_CIFAR10,
                                      VGG_CIFAR10)

#needed for using nn-Meter to make latency predictions and measurements
import nn_meter

# enable GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
print('Using PyTorch version:', torch.__version__, ' Device:', device, '\n')

print('Using nn-Meter version:', nn_meter.__version__, '\n')

#let user select the hardware predictor model
predictors = nn_meter.list_latency_predictors()
print("Select from the list of available architecture predictor models (enter a value 1-4):")
for idx, p in enumerate(predictors, start = 1):
    print(f"{idx}: [Predictor] {p['name']}: version={p['version']}")

predictor_choice = input()
try:
    predictor_index = int(predictor_choice)
    if 1 <= predictor_index <= 4:
        predictor_name = predictors[predictor_index - 1]['name']
    else:
        print("Invalid selection! Please restart and choose a valid option.")
        sys.exit()
except ValueError:
    print("Invalid input. Aborting program.")
    sys.exit()

print(f"You have selected device architecture predictor model: {predictor_name}\n")
predictor = nn_meter.load_latency_predictor(predictor_name)

#let the user select the convolutional kernel size
kernel_classes = {
    '1': ('kernel_3x3_model_classes', kernel_3x3_model_classes),
    '2': ('kernel_5x5_model_classes', kernel_5x5_model_classes),
    '3': ('kernel_7x7_model_classes', kernel_7x7_model_classes),
}

print("\nSelect convolution kernel size for future CNN architecture selection:")
print("1: 3x3 Kernel Models")
print("2: 5x5 Kernel Models")
print("3: 7x7 Kernel Models")

kernel_choice = input()

if (kernel_choice == '1'):
    kernel_size = '3x3'
elif (kernel_choice == '2'):
    kernel_size = '5x5'
elif (kernel_choice == '3'):
    kernel_size = '7x7'
    
try:
    kernel_index = kernel_classes.get(kernel_choice)
    if kernel_index is None:
        print("Invalid selection! Please restart and choose a valid option.")
        sys.exit()
    kernel_class_name, model_classes = kernel_index
except ValueError:
    print("Invalid input. Aborting program.")
    sys.exit()

print(f"You have selected the kernel class: {kernel_class_name}\n")

#let the user select the CNN architecture
classes = inspect.getmembers(model_classes, inspect.isclass)
nn_model_names = [name for name, obj in classes if obj.__module__ == kernel_class_name]

print("\nNow, select the NN architecture to be trained and tested (enter index value):")
for idx, name in enumerate(nn_model_names, start=1):
    print(f"{idx}: {name}")

nn_model_choice = input()
try:
    model_index = int(nn_model_choice)
    if 1 <= model_index <= len(nn_model_names):
        model_name = nn_model_names[model_index - 1]
        model_class = getattr(model_classes, model_name)
        model = model_class()
    else:
        print("Invalid selection! Please restart and choose a valid option.")
        sys.exit()
except (ValueError, AttributeError):
    print("Invalid input. Aborting program.")
    sys.exit()

print(f"You have selected DNN architecture model: {model.__class__.__name__}\n")

print(f"Now predicting feed-forward latency for {model.__class__.__name__} architecture with kernel size {kernel_size} on {predictor_name}...\n")

#train loader for CIFAR10 image data
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('data', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])),
    batch_size=64, shuffle=True)    

#test loader for CIFAR10 image data
test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])),
    batch_size=64, shuffle=True)
    
#defining optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

#defining loss function
criterion = nn.NLLLoss()

#training the selected CNN architecture
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)  # Pass the raw 32x32x3 images
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

#testing the selected CNN architecture
def test():  
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)  #pass the CIFAR10 32x32x3 image data
            test_loss += criterion(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%) '.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print('Training Time: %.3f' % (end-start))
    
    #define latency and display the resulting latency measurement
    #input data shape must be manually defined for nn-Meter
    latency = predictor.predict(model, model_type = "torch", input_shape = (1, 3, 32, 32))
    print(f"\nLatency prediction for {model.__class__.__name__} architecture with {kernel_size} kernel size tested on {predictor_name} target hardware is: {latency} ms")
    
# main
if __name__ == '__main__': 
    start = time.time()
    for epoch in range(1, 10):
        train(epoch)
    end = time.time()

    test() 
    
    # Save the model
    torch.save(model.state_dict(), "mnist_mlp.pt")


