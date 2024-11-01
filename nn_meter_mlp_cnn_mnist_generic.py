#Contributors: Dr. Anagnostopoulos
#              Jake Horio
#              Mason Mac
#Last Modified: 10/31/24
#Description: Main code module that uses the nn-Meter module tool to measure the 
#             forward-pass latency on DNNs running on targeted new-edge cores

import sys
import inspect
import time
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms

#classes of DNN architectures
import model_classes
from model_classes import MLP, CNN

#needed for using nn-Meter to make latency predictions and measurements
import nn_meter

#needed for creating the confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# enable GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
print('Using PyTorch version:', torch.__version__, ' Device:', device, '\n')

print('Using nn-Meter version:', nn_meter.__version__, '\n')

###################let user select the predictor model architecture to be simulated####################
predictors = nn_meter.list_latency_predictors()
print("Select from the list of avaiable architecture predictor models (enter a value 1-4):")
x = 1
for p in predictors:
    print(f"{x}: [Predictor] {p['name']}: version={p['version']}")
    x = x + 1
predictor_choice = input()
try:
    select = int(predictor_choice)
    if select in [1, 2, 3, 4]:
        if select == 1:
            predictor_name = "cortexA76cpu_tflite21"
        if select == 2:
            predictor_name = "adreno640gpu_tflite21"
        if select == 3:
            predictor_name = "adreno630gpu_tflite21"
        if select == 4:
            predictor_name = "myriadvpu_openvino2019r2" 
    else:
        print("Not a valid selection! Please restart the program and") 
        print("select options 1-4 per the predictor model options")
        sys.exit()
except ValueError:
    print("Not a valid input. Aborting the program")   
print(f"You have selected device architecture predictor model: {predictor_name}\n")   
predictor = nn_meter.load_latency_predictor(predictor_name)

#################################let user select the DNN architecture#################################
classes = inspect.getmembers(model_classes, inspect.isclass)
nn_model_names = [name for name, obj in classes if obj.__module__ == 'model_classes']
print("\nNow, select the NN architecture to be trained and tested (enter index value):")
for idx, name in enumerate(nn_model_names, start=1):
    print(f"{idx}: {name}")
nn_model_choice = input()
try:
    select = int(nn_model_choice)
    if select in [1, 2]:
        if select == 1:
            model = CNN()
        if select == 2:
            model = MLP()
    else:
        print("Not a valid selection! Please restart the program and") 
        print("select an index value per the CNN model options")
        sys.exit()
except ValueError:
    print("Not a valid input. Aborting the program")
print(f"You have selected DNN architecture model: {model.__class__.__name__}\n")

# Load the training data
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                     transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,))
                        ])),
    batch_size=64, shuffle=True)    

# Load the test data
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,))
                        ])),
    batch_size=64, shuffle=True)
    
# Define the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Define the loss function
criterion = nn.NLLLoss()

# Train the model
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 784)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

# Test the model
def test():
    
    #arrays for the target and prediction values for the confusion matrix
    y_test = []
    predictions = []
    
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(-1, 784)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            
            #filling both the target and prediction arrays for the confusion matrix
            y_test.extend(target.numpy())
            predictions.extend(pred.numpy())

    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%) '.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print('Training Time: %.3f' % (end-start))
    
    #display the confusion matrix data
    conf_matrx = confusion_matrix(y_test, predictions)
    ConfusionMatrixDisplay(conf_matrx).plot()
    
    #define latency and display the resulting latency measurement
    #input data shape must be manually defined for nn-Meter
    #tensor is shaped to (1, 784) for the MNIST digit dataset
    latency = predictor.predict(model, model_type = "torch", input_shape = (1, 784))
    print(f"\nPredicted latency for device architecture model {predictor_name} is : {latency} ms")
    
# main
if __name__ == '__main__': 
    
    # Run the training loop
    # This is the loop you have to time
    start = time.time()
    for epoch in range(1, 10):
        train(epoch)
    end = time.time()

    test() 
    
    # Save the model
    torch.save(model.state_dict(), "mnist_mlp.pt")


