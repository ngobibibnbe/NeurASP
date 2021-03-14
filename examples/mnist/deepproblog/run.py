import sys
import os
cwd = os.getcwd()
print("--------------"+cwd)

from train import train_model
from data_loader import load
from mnist import test_MNIST, MNIST_Net, neural_predicate
from model import Model
from optimizer import Optimizer
from network import Network
import torch
import random
random.seed(0)
torch.manual_seed(0)
import numpy as np
np.random.seed(0)

queries = load('train_data.txt')

with open('addition.pl') as f:
    problog_string = f.read()

@torch.no_grad()
def init_weights(m):
    print(m)
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        m.weight.fill_(1.0)
        print(m.weight)

network = MNIST_Net()
net = Network(network, 'mnist_net', neural_predicate)
net.optimizer = torch.optim.Adam(network.parameters(),lr = 0.001)
model = Model(problog_string, [net], caching=False)
optimizer = Optimizer(model, 2)

train_model(model,queries, 1, optimizer,test_iter=1000,test=test_MNIST,snapshot_iter=10000)
