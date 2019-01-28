# The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. 

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# the steps to load the train and test data set is from Pytorch tutorials
# depends on downloaded or not ---------------------
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding =4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# device to use
use_cuda = torch.cuda.is_available()
this_device = torch.device("cuda" if use_cuda else "cpu")

class ConvNet(nn.Module):

    # tune all the dropout rate -----------------------------
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.1) 
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2) 
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Dropout(0.3) 
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.layer8 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5) 
        )
        self.fc1 = nn.Linear(4*4* 64, 500)
        self.fc2 = nn.Linear(500, 10)    

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)

        out = out.reshape(out.size(0), -1)
        #out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

my_cnn = ConvNet().to(this_device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(my_cnn.parameters(), lr=0.0001)

for epoch in range(30):  # loop over the dataset multiple times

    if(epoch>6):
	for group in optimizer.param_groups:
		for p in group['params']:
			state = optimizer.state[p]
			if(state['step']>=1024):
				state['step'] = 1000

    for i, (data, target) in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data.to(this_device), target.to(this_device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = my_cnn(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print("Running epoch: ", epoch)

print('Finished Training')

# testing ---------------------------------------------------
correct = 0
total = 0
with torch.no_grad():
    for i, (data, target) in enumerate(testloader):
        images, labels = data.to(this_device), target.to(this_device)
        outputs = my_cnn(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for i, (data, target) in enumerate(testloader):
        images, labels = data.to(this_device), target.to(this_device)
        outputs = my_cnn(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

# save and load only the model parameters (recommended). 
torch.save(my_cnn.state_dict(), './hw3_params.ckpt') 


