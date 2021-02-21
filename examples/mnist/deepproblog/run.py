import sys
#sys.path.append('/content/drive/MyDrive/Projets/deepproblog/')
import os
cwd = os.getcwd()
print("--------------"+cwd)

from train import train_model
from data_loader import load
#sys.path.append('examples/NIPS/MNIST')
from mnist import test_MNIST, MNIST_Net, neural_predicate
from model import Model
from optimizer import Optimizer
from network import Network
import torch

#os.chdir('/content/drive/MyDrive/Projets/deepproblog/')

#sys.path.append('C:/Users/sophie/Desktop/iCS/reseacrch project/deepproblog/examples/NIPS/MNIST/single_digit')
queries = load('train_data.txt')

with open('addition.pl') as f:
    problog_string = f.read()


network = MNIST_Net()
net = Network(network, 'mnist_net', neural_predicate)
net.optimizer = torch.optim.Adam(network.parameters(),lr = 0.001)
model = Model(problog_string, [net], caching=False)
optimizer = Optimizer(model, 2)

train_model(model,queries, 1, optimizer,test_iter=1000,test=test_MNIST,snapshot_iter=10000)
