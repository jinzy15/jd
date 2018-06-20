import torch
import random
import transformer.Modules as Modules
from torch import optim
import utils.Lang as Lang
from dataset.groupSet import groupSet
import torch.nn as nn
from Config import *
import transformer.Modules as Modules
from transformer.Models.LHRED import LHRED
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader


mylang = Lang.npLang('all','data/clean_group.npy')
# mylang.prepareData()
# mylang.saveLang('./utils/npLang')
mylang.loadLang('./utils/npLang')
print('all words',mylang.n_words)

mydata = groupSet(mylang)
mydata.loadfnp('data/clean_group.npy')

input_size = mylang.n_words
output_size = mylang.n_words

trainloader = DataLoader(mydata)
testloader = DataLoader(mydata)

encoder = Modules.SessionEncoderRNN(input_size, hidden_size,batch_size).to(device)
decoder = Modules.DecoderRNN(hidden_size, output_size).to(device)
context = Modules.ContextRNN(hidden_size,hidden_size).to(device)

if (use_histmodel):
    encoder.load_state_dict(torch.load(histmodel_path+'_encoder.pkl'))
    decoder.load_state_dict(torch.load(histmodel_path+'_decoder.pkl'))
    context.load_state_dict(torch.load(histmodel_path+'_context.pkl'))

mymodel = LHRED(mydata,trainloader,testloader,encoder,decoder,context)

if(is_train):
    # mymodel.trainEpoch(2,1)
    mymodel.trainIters(2000, 10)

if(is_evaluate):
    mymodel.evaluateRandomly(10)


