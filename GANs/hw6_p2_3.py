import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

import os

from hw6_gan import generator, discriminator
from hw6_func import plot

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

batch_size = 128

transform_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

testset = torchvision.datasets.CIFAR10(root='./', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=32)
testloader = enumerate(testloader)

#print("Disc model with generator")
#model_disc = torch.load('discriminator.model')

print("Disc model without generator")
model_disc = torch.load('hw6_cifar10_disc.model')

model_disc.cuda()
model_disc.eval()

model_disc = torch.nn.DataParallel(model_disc)
cudnn.benchmark = True

batch_idx, (X_batch, Y_batch) = testloader.__next__()

X_batch = Variable(X_batch,requires_grad=True).cuda()
Y_batch_alternate = (Y_batch + 1)%10
Y_batch_alternate = Variable(Y_batch_alternate).cuda()
Y_batch = Variable(Y_batch).cuda()

X = X_batch.mean(dim=0)
X = X.repeat(batch_size,1,1,1)

Y = torch.arange(batch_size).type(torch.int64)
Y = Variable(Y).cuda()

layers = [4, 8]
for layer in layers:
    lr = 0.1
    weight_decay = 0.001
    for i in range(200):
        output = model_disc(X, layer)

        loss = -output[torch.arange(batch_size).type(torch.int64),torch.arange(batch_size).type(torch.int64)]
        gradients = torch.autograd.grad(outputs=loss, inputs=X,
                                grad_outputs=torch.ones(loss.size()).cuda(),
                                create_graph=True, retain_graph=False, only_inputs=True)[0]

        prediction = output.data.max(1)[1] # first column has actual prob.
        accuracy = ( float( prediction.eq(Y.data).sum() ) /float(batch_size))*100.0
        print(i,accuracy,-loss)

        X = X - lr*gradients.data - weight_decay*X.data*torch.abs(X.data)
        X[X>1.0] = 1.0
        X[X<-1.0] = -1.0

    ## save new images
    samples = X.data.cpu().numpy()
    samples += 1.0
    samples /= 2.0
    samples = samples.transpose(0,2,3,1)

    fig = plot(samples[0:100])

#    filename = 'visualization/max_features_w_gen_{}.png'.format(layer)
    filename = 'visualization/max_features_wo_gen_{}.png'.format(layer)

    plt.savefig(filename, bbox_inches='tight')
plt.close(fig)










