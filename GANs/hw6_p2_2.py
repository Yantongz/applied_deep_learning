import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.autograd.variable as Variable

import torchvision
import torchvision.transforms as transforms

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import argparse
from hw6_func import plot

#parser = argparse.ArgumentParser(description='Hw6: GAN on CIFAR10 disc models')
#parser.add_argument('--with_gen', help = 'which model to use')
#args = parser.parse_args()

batch_size = 128

transform_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

testset = torchvision.datasets.CIFAR10(root='./', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=32)
testloader = enumerate(testloader)


#if args.with_gen==1:
print("Disc model with generator")
model_disc = torch.load('discriminator.model')

#if args.with_gen==0:
#print("Disc model with generator")
#model_disc = torch.load('discriminator.model')

model_disc.cuda()
model_disc.eval()
print('Load and eval discriminator model ...')

model_disc = torch.nn.DataParallel(model_disc)
cudnn.benchmark = True

# load in a model and a batch of images
batch_idx, (X_batch, Y_batch) = next(testloader)

X_batch = Variable(X_batch,requires_grad=True).cuda()
Y_batch_alternate = (Y_batch + 1)%10
Y_batch_alternate = Variable(Y_batch_alternate).cuda()
Y_batch = Variable(Y_batch).cuda()

# calculate the mean image and make 10 copies (number of classes).
X = X_batch.mean(dim=0)
X = X.repeat(10,1,1,1)

# Make a unique label for each copy.
Y = torch.arange(10).type(torch.int64)
Y = Variable(Y).cuda()

lr = 0.1
weight_decay = 0.001

# The loss function for each class will be used to repeatedly modify an image such that it maximizes the output.

for i in range(200):
    _, output = model_disc(X)

    loss = -output[torch.arange(10).type(torch.int64),torch.arange(10).type(torch.int64)]
    gradients = torch.autograd.grad(outputs=loss, inputs=X,
                              grad_outputs=torch.ones(loss.size()).cuda(),
                              create_graph=True, retain_graph=False, only_inputs=True)[0]

    prediction = output.data.max(1)[1] # first column has actual prob.
    accuracy = ( float( prediction.eq(Y.data).sum() ) /float(10.0))*100.0
    print(i,accuracy,-loss)

    # the original image X is modified by this gradient to maximize the output. 
    X = X - lr*gradients.data - weight_decay*X.data*torch.abs(X.data)
    X[X>1.0] = 1.0
    X[X<-1.0] = -1.0

## save new images
samples = X.data.cpu().numpy()
samples += 1.0
samples /= 2.0
samples = samples.transpose(0,2,3,1)

fig = plot(samples)
#if args.with_gen==1:
#    filename = 'visualization/max_class_disc_gen.png'

#if args.with_gen==0:
filename = 'visualization/max_class_w_gen.png'
 
plt.savefig(filename, bbox_inches='tight')
plt.close(fig)

print('Saved synthetic images maximizing class output ...')

