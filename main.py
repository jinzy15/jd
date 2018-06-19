# coding: utf-8

from __future__ import unicode_literals, print_function, division
from utils.Lang import Lang
from Config import *
from dataset.pairSet import pairSet
import transformer.Modules as Modules
import transformer.Models as Models
from torch.utils.data import DataLoader

from io import open
import unicodedata
import string
import re
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import os
import numpy as np

mylang = Lang('all',pair_file)
pairs = mylang.prepareData(False)
mydata = pairSet(pairs,mylang)

encoder1 = Modules.EncoderRNN(mylang.n_words, hidden_size).to(device)
attn_decoder1 = Modules.AttnDecoderRNN(hidden_size, mylang.n_words, dropout_p=0.1).to(device)

if (use_histmodel):
    encoder1.load_state_dict(torch.load(histmodel_path+'_encoder.pkl'))
    attn_decoder1.load_state_dict(torch.load(histmodel_path+'_decoder.pkl'))

trainloader = DataLoader(mydata)
testloader = DataLoader(mydata)

mymodel = Models.EDModels(mydata,trainloader,testloader,encoder1,attn_decoder1)

if(is_train):
    mymodel.trainEpoch(2,10)

if(is_evaluate):
    mymodel.evaluateRandomly()
    output_words, attentions = mymodel.evaluate("你好")
    print(output_words)


