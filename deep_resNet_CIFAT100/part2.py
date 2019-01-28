import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models


# the steps to load the train and test data set is from Pytorch tutorials
# depends on downloaded or not ---------------------
transform_train = transforms.Compose([ # have to up-sample
    transforms.RandomSizedCrop(224),  # input size for the pretrained model is 224*224
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

transform_test = transforms.Compose([
    transforms.Scale(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=8)


# device to use
use_cuda = torch.cuda.is_available()
this_device = torch.device("cuda" if use_cuda else "cpu")

# load model
resnet_18 = models.resnet18(pretrained=True)
num_ftrs = resnet_18.fc.in_features
resnet_18.fc = nn.Linear(num_ftrs, 100)

resnet_18 = resnet_18.to(this_device)

# resNet relies on residual modules via identity mappings, high learning rate is possible.

criterion = nn.CrossEntropyLoss()

for para in list(resnet_18.parameters())[:-2]:
    para.requires_grad=False 

optimizer = optim.SGD(resnet_18.parameters(), lr=1e-4, momentum = 0.9)


total_step = len(trainloader)
loss_list = []

num_epochs = 20

for epoch in range(num_epochs):  # loop over the dataset multiple times

    for i, (data, target) in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data.to(this_device), target.to(this_device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = resnet_18(inputs)
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
        outputs = resnet_18(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on test data: %d %%' % (100 * correct / total))


# save and load only the model parameters (recommended). 
torch.save(resnet_18.state_dict(), './hw4_p2_params.ckpt') 












