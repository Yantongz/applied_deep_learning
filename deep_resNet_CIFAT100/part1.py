# CIFAR 100 has 100 classes containing 600 images each. 
# There are 500 training images and 100 testing images per class. 

# The 100 classes in the CIFAR-100 are grouped into 20 superclasses. Each image comes with a "fine" label (the class to which it belongs) and a "coarse" label (the superclass to which it belongs).

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# the steps to load the train and test data set is from Pytorch tutorials
# depends on downloaded or not ---------------------

# test data augmentation methods ---------------------
transform_train = transforms.Compose([
    # tune augmentation methods
    transforms.RandomCrop(32, padding = 4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

# tune batch size -----------------------------------

trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=125,
                                          shuffle=True, num_workers=12)

testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=125,
                                         shuffle=False, num_workers=12)

# device to use
use_cuda = torch.cuda.is_available()
this_device = torch.device("cuda" if use_cuda else "cpu")


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inchannel, outchannel, stride =1):
        super(BasicBlock, self).__init__()
	# all filter size: 3x3, all padding = 1 --------------------
        self.conv1 = nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outchannel)
        self.conv2 = nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outchannel)

        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != self.expansion*outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, self.expansion*outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*outchannel)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):

    # (each) resNet model contains multiple layers (convN_x), each layer contains multiple residual block
    def __init__(self, block, num_blocks, num_classes = 100): # see first line
        super(ResNet, self).__init__()
        self.inchannel = 32

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.drop1 = nn.Dropout(0.3, inplace = True)   # tune

        self.conv2x = self._make_layer(block, 32, num_blocks[0], stride = 1)
        self.conv3x = self._make_layer(block, 64, num_blocks[1], stride = 2)
        self.conv4x = self._make_layer(block, 128, num_blocks[2], stride = 2)
        self.conv5x = self._make_layer(block, 256, num_blocks[3], stride = 2)

        self.fc = nn.Linear(256*block.expansion, num_classes)

    # function to make multiple basic blocks in one layer
    def _make_layer(self, block, outchannel, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks -1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, outchannel,stride))
            self.inchannel = outchannel*block.expansion
        return nn.Sequential(*layers)


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.drop1(out)
        out = self.conv2x(out)
        out = self.conv3x(out)
        out = self.conv4x(out)
        out = self.conv5x(out)

        out = F.max_pool2d(out,kernel_size = 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


my_resNet = ResNet(BasicBlock, [2,4,4,2]).to(this_device)

# resNet relies on residual modules via identity mappings, high learning rate is possible.

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(my_resNet.parameters(), lr=0.001)

total_step = len(trainloader)
loss_list = []

num_epochs = 120

for epoch in range(num_epochs):  # loop over the dataset multiple times

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
        outputs = my_resNet(inputs)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())

        loss.backward()
        optimizer.step()

	# Track the accuracy
        total_train = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct_train = (predicted == labels).sum().item()
        
        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),(correct_train/total_train) * 100))


print('Finished Training')

# testing ---------------------------------------------------
correct = 0
total = 0
with torch.no_grad():
    for i, (data, target) in enumerate(testloader):
        images, labels = data.to(this_device), target.to(this_device)
        outputs = my_resNet(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on test data: %d %%' % (100 * correct / total))

# save and load only the model parameters (recommended). 
torch.save(my_resNet.state_dict(), './hw4_p1_params.ckpt') 









