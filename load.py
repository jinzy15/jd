import torch
import random
import transformer.Modules as Modules
from torch import optim
import utils.Lang as Lang
from dataset.groupSet import groupSet
import torch.nn as nn
from Config import *
import transformer.Modules as Modules
import transformer.Models as Models
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

mylang = Lang.npLang('all','data/clean_group.npy')
mylang.prepareData()
mydata = groupSet(mylang)
mydata.loadfnp('data/clean_group.npy')

input_size = mylang.n_words
output_size = mylang.n_words
hidden_size = 128

trainloader = DataLoader(mydata)
testloader = DataLoader(mydata)

encoder = Modules.SessionEncoderRNN(mylang.n_words, hidden_size,batch_size).to(device)
decoder = Modules.DecoderRNN(hidden_size, mylang.n_words).to(device)
context = Modules.ContextRNN(hidden_size,hidden_size).to(device)








