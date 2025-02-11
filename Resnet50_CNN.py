import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import resnet50
from tqdm import tqdm

# Device agnostic code.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# Define Data Transformation
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])

batch_size = 64

# Load train dataset
train_dataset = torchvision.datasets.CIFAR10 (root='./data', train= True, transform= transform, download = True)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# Load test dataset
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# Load pre trained ResNet 50
model = resnet50(pretrained=True)
model.fc = nn.Linear(2048,10) # As CIFAR-10 has 10 classes
model = model.to(device)

# Define Loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

#Setup Training function
def train_function(model, trainloader, loss, optimizer, epochs=5):
    model.train()
    for epoch in tqdm(range(epochs)):
        running_loss = 0
        correct = 0
        total = 0

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        print(f"Epoch {epoch+1}: Loss: {running_loss/len(trainloader):.4f}, Accuracy: {100*correct/total:.2f}%")

# Train the model
train_function(model=model,
               trainloader=train_dataloader,
               loss=loss_fn,
               optimizer=optimizer)

torch.save(model.state_dict(),'resnet50_cifar10.pth')
print('Model weights saved successfully')
#model.load_state_dict(torch.load('resnet50_cifar10.pth'))

# Testing function

def test_model(model, testloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _,predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    print(f'Test Accuracy: {100* correct /total:.2f}%')

#Evaluate the model
test_model(model=model,
          testloader=test_dataloader) #Test function