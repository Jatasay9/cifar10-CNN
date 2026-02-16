# import torch
# import torchvision
# import torchvision.transforms as transforms
# import torch.nn as nn
# import torch.optim as optim




# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Device:", device)

# transform_train = transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomCrop(32, padding=4),
#     transforms.ToTensor()
# ])

# transform_test = transforms.Compose([
#     transforms.ToTensor()
# ])


# trainset = torchvision.datasets.CIFAR10(
#     root='./data',
#     train=True,
#     download=True,
#     transform=transform_train
# )


# trainloader = torch.utils.data.DataLoader(
#     trainset,
#     batch_size=64,
#     shuffle=True
# )

# images, labels = next(iter(trainloader))
# print("Image batch shape:", images.shape)



# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super(SimpleCNN, self).__init__()

#         self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
#         self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

#         self.pool = nn.MaxPool2d(2,2)
#         self.relu = nn.ReLU()

#         self.fc1 = nn.Linear(32 * 8 * 8, 10)
        


#     def forward(self, x):
#         x = self.pool(self.relu(self.conv1(x)))
#         x = self.pool(self.relu(self.conv2(x)))
#         x = x.view(x.size(0), -1)
        
#         x = self.fc1(x)
#         return x


# model = SimpleCNN().to(device)

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# for epoch in range(10):
#     running_loss = 0.0
#     correct = 0
#     total = 0

#     model.train()

#     for images, labels in trainloader:
#         images, labels = images.to(device), labels.to(device)

#         optimizer.zero_grad()

#         outputs = model(images)
#         loss = criterion(outputs, labels)

#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()

#         _, predicted = torch.max(outputs, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

#     print(f"Epoch {epoch+1}, Loss: {running_loss:.3f}, Accuracy: {100 * correct/total:.2f}%")
# testset = torchvision.datasets.CIFAR10(
#     root='./data',
#     train=False,
#     download=True,
#     transform=transform_test
# )


# testloader = torch.utils.data.DataLoader(
#     testset,
#     batch_size=64,
#     shuffle=False
# )
# transform_train = transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomCrop(32, padding=4),
#     transforms.ToTensor()
# ])

# transform_test = transforms.Compose([
#     transforms.ToTensor()
# ])





# model.eval()
# correct = 0
# total = 0

# with torch.no_grad():
#     for images, labels in testloader:
#         images, labels = images.to(device), labels.to(device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print(f"Test Accuracy: {100 * correct/total:.2f}%")


import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Transforms
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor()
])

transform_test = transforms.Compose([
    transforms.ToTensor()
])

# Load CIFAR-10 subset (10k samples)
trainset_full = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform_train
)

subset_indices = list(range(10000))
trainset = torch.utils.data.Subset(trainset_full, subset_indices)

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=64,
    shuffle=True
)

# Test set
testset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform_test
)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=64,
    shuffle=False
)

# Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

        self.pool = nn.MaxPool2d(2,2)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(32 * 8 * 8, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

model = SimpleCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
train_acc_list = []

for epoch in range(10):
    model.train()
    correct = 0
    total = 0
    running_loss = 0.0

    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_acc = 100 * correct / total
    train_acc_list.append(epoch_acc)

    print(f"Epoch {epoch+1}, Loss: {running_loss:.3f}, Accuracy: {epoch_acc:.2f}%")

# Evaluation
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
print(f"Test Accuracy: {test_accuracy:.2f}%")

# Plot training curve
plt.plot(train_acc_list)
plt.xlabel("Epoch")
plt.ylabel("Train Accuracy (%)")
plt.title("Training Accuracy Curve")
plt.savefig("training_curve.png")
plt.show()
