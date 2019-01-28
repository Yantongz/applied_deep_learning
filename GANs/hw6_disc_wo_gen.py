# define the models for disc without gen
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import torch.autograd.variable as Variable

import torchvision
import torchvision.transforms as transforms

from hw6_gan import discriminator

# transformation

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.7, 1.0), ratio=(1.0,1.0)),
    transforms.ColorJitter(
        brightness=0.1*torch.randn(1),
        contrast=0.1*torch.randn(1),
        saturation=0.1*torch.randn(1),
        hue=0.1*torch.randn(1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

batch_size = 128

trainset = torchvision.datasets.CIFAR10(root='./', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=32)

testset = torchvision.datasets.CIFAR10(root='./', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=32)

model_disc =  discriminator()
model_disc.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_disc.parameters(), lr=0.0001)

total_step = len(trainloader)
learning_rate = 0.0001
 
for epoch in range(100):  # loop over the dataset multiple times

    if(epoch>6): 
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                if('step' in state and state['step']>=1024):
                    state['step'] = 1000
    if(epoch==50): 
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate/10.0 
    if(epoch==75):
        for param_group in optimizer.param_groups: 
            param_group['lr'] = learning_rate/100.0
 
    for batch_idx, (X_train_batch, Y_train_batch) in enumerate(trainloader):

        if(Y_train_batch.shape[0] < batch_size):
            continue
 
        X_train_batch = Variable(X_train_batch).cuda()
        Y_train_batch = Variable(Y_train_batch).cuda() 
        _, output = model_disc(X_train_batch) 
 
        loss = criterion(output, Y_train_batch)
        #loss_list.append(loss.item())
        optimizer.zero_grad()
 
        loss.backward() 
        optimizer.step()

        if (batch_idx+1)%200 ==0:
            print('Running epoch: [{}/{}], step[{}/{}], Loss: {:.4f}'.format(epoch+1, 100, batch_idx+1, total_step, loss.item()))

print('Finished Training')

# testing ---------------------------------------------------
correct = 0
total = 0
with torch.no_grad():
    for i, (data, target) in enumerate(testloader):
        images = Variable(data).cuda()
        labels = Variable(target).cuda()
        _, outputs = model_disc(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))


torch.save(model_disc,'./hw6_cifar10_disc.model')


