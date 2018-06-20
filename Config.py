import torch
# import matplotlib.pyplot as plt

# plt.switch_backend('agg')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = torch.cuda.is_available()
description = 'debug_mode'
is_train = True
is_evaluate = True
use_histmodel = False
histmodel_path = 'train_fruit/'+description
pair_file = 'data/chat_pair.txt'
teacher_forcing_ratio = 0.5
hidden_size = 128
EOS_token = 2
SOS_token = 1
PAD_token = 0
UNK_token = 3
batch_size = 30
MAX_LENGTH = 30