import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import torch.autograd.variable as Variable

import torchvision
import torchvision.transforms as transforms

import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import numpy as np

# for gen and disc model, other func
from hw6_gan import discriminator, generator
from hw6_func import plot, calc_gradient_penalty, grad_update_thres

batch_size = 128

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

trainset = torchvision.datasets.CIFAR10(root='./', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=32)

testset = torchvision.datasets.CIFAR10(root='./', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=32)

print('load the models')
# initialize
aD =  discriminator()
aD.cuda()

aG = generator()
aG.cuda()

learning_rate = 0.0001
optimizer_g = torch.optim.Adam(aG.parameters(), lr=0.0001, betas=(0,0.9))
optimizer_d = torch.optim.Adam(aD.parameters(), lr=0.0001, betas=(0,0.9))

criterion = nn.CrossEntropyLoss()

# batch size is 100 and there are 10 examples for each class
n_classes = 10
n_z = 100

# tune this param ---------------
gen_train = 1

# a random batch of noise for the generator; ACGAN
np.random.seed(352)
label = np.asarray(list(range(10))*10)
noise = np.random.normal(0,1,(100,n_z))
label_onehot = np.zeros((100,n_classes))
label_onehot[np.arange(100), label] = 1
noise[np.arange(100), :n_classes] = label_onehot[np.arange(100)]
noise = noise.astype(np.float32)

save_noise = torch.from_numpy(noise)
save_noise = Variable(save_noise).cuda()

def training_process(epoch):

	loss1, loss2, loss3, loss4, loss5 = [], [], [], [], []
	acc1=[]

	aG.train()
	aD.train()
	for batch_idx, (X_train_batch, Y_train_batch) in enumerate(trainloader):

		if(Y_train_batch.shape[0] < batch_size):
			continue

		# train G
		if((batch_idx%gen_train)==0):
			for p in aD.parameters():
				p.requires_grad_(False)

		aG.zero_grad()

		label = np.random.randint(0,n_classes,batch_size)
		noise = np.random.normal(0,1,(batch_size,n_z))
		label_onehot = np.zeros((batch_size,n_classes))
		label_onehot[np.arange(batch_size), label] = 1
		noise[np.arange(batch_size), :n_classes] = label_onehot[np.arange(batch_size)]
		noise = noise.astype(np.float32)
		noise = torch.from_numpy(noise)
		noise = Variable(noise).cuda()
		fake_label = Variable(torch.from_numpy(label)).cuda()

		fake_data = aG(noise)
		gen_source, gen_class  = aD(fake_data)

		gen_source = gen_source.mean()
		gen_class = criterion(gen_class, fake_label)

		gen_cost = -gen_source + gen_class
		gen_cost.backward()

		grad_update_thres(optimizer_g)
		optimizer_g.step()

		# train D
		if((batch_idx%gen_train)==0):
                    for p in aD.parameters():
                        p.requires_grad_(True)

		aD.zero_grad()

		# train discriminator with input from generator
		label = np.random.randint(0,n_classes,batch_size)
		noise = np.random.normal(0,1,(batch_size,n_z))
		label_onehot = np.zeros((batch_size,n_classes))
		label_onehot[np.arange(batch_size), label] = 1
		noise[np.arange(batch_size), :n_classes] = label_onehot[np.arange(batch_size)]
		noise = noise.astype(np.float32)
		noise = torch.from_numpy(noise)
		noise = Variable(noise).cuda()
		fake_label = Variable(torch.from_numpy(label)).cuda()
		with torch.no_grad():
		    fake_data = aG(noise)

		disc_fake_source, disc_fake_class = aD(fake_data)

		disc_fake_source = disc_fake_source.mean()
		disc_fake_class = criterion(disc_fake_class, fake_label)

		# train discriminator with input from the discriminator
		real_data = Variable(X_train_batch).cuda()
		real_label = Variable(Y_train_batch).cuda()

		disc_real_source, disc_real_class = aD(real_data)

		prediction = disc_real_class.data.max(1)[1]
		accuracy = ( float( prediction.eq(real_label.data).sum() ) /float(batch_size))*100.0

		disc_real_source = disc_real_source.mean()
		disc_real_class = criterion(disc_real_class, real_label)

		gradient_penalty = calc_gradient_penalty(aD,real_data,fake_data)

		disc_cost = disc_fake_source - disc_real_source + disc_real_class + disc_fake_class + gradient_penalty
		disc_cost.backward()

		grad_update_thres(optimizer_d)
		optimizer_d.step()

		# within the training loop
		loss1.append(gradient_penalty.item())
		loss2.append(disc_fake_source.item())
		loss3.append(disc_real_source.item())
		loss4.append(disc_real_class.item())
		loss5.append(disc_fake_class.item())
		acc1.append(accuracy)
		if((batch_idx%50)==0):
		    print('\nEpoch:', epoch+1, 'batch index:', batch_idx, 'loss 1-5:', "%.2f" % np.mean(loss1), "%.2f" % np.mean(loss2), "%.2f" % np.mean(loss3), "%.2f" % np.mean(loss4), "%.2f" % np.mean(loss5),', train accuracy:', "%.2f" % np.mean(acc1))


def testing_process(epoch):

    # Test the model
    aD.eval()
    with torch.no_grad():
        test_accu = []
        for batch_idx, (X_test_batch, Y_test_batch) in enumerate(testloader):
            X_test_batch, Y_test_batch= Variable(X_test_batch).cuda(),Variable(Y_test_batch).cuda()

            with torch.no_grad():
                _, output = aD(X_test_batch)

            prediction = output.data.max(1)[1] # first column has actual prob.
            accuracy = ( float( prediction.eq(Y_test_batch.data).sum() ) /float(batch_size))*100.0
            test_accu.append(accuracy)
            accuracy_test = np.mean(test_accu)
    print('\nEpoch:', epoch+1, ', testing accuracy: ',"%.2f" % accuracy_test)


def eval_gen_noise(epoch):

    ### save output
    with torch.no_grad():
        aG.eval()
        samples = aG(save_noise)
        samples = samples.data.cpu().numpy()
        samples += 1.0
        samples /= 2.0
        samples = samples.transpose(0,2,3,1)
        aG.train()

    fig = plot(samples)
    plt.savefig('output/%s.png' % str(epoch).zfill(3), bbox_inches='tight')
    plt.close(fig)


for epochi in range(200):
    start_time = time.time()
    training_process(epochi)
    testing_process(epochi)
    print("\nTime for one epoch:", epochi+1, time.time()-start_time)

    eval_gen_noise(epochi)

    if(((epochi+1)%1)==0):
        torch.save(aG,'./tempG.model')
        torch.save(aD,'./tempD.model')

torch.save(aG,'./generator.model')
torch.save(aD,'./discriminator.model')


