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

for item in mydata[0]:
    print(item)
# for session in mydata:
#     criterion = nn.NLLLoss()
#     in_session = session[0:-1]
#     tar_session = session[1:]
#     hidden = torch.zeros(1, in_session.shape[0], hidden_size, device=device)
#     context_hidden = torch.zeros(1,1,hidden_size,device = device)
#     encode = Modules.SessionEncoderRNN(input_size,hidden_size,30)
#     context = Modules.ContextRNN(hidden_size,hidden_size)
#     decode = Modules.DecoderRNN(hidden_size,output_size)
#     learning_rate = 0.01
#     encoder_optimizer = optim.SGD(encode.parameters(), lr=learning_rate)
#     context_optimizer = optim.SGD(context.parameters(),lr= learning_rate)
#     decoder_optimizer = optim.SGD(decode.parameters(), lr=learning_rate)
#
#     in_session,session_hidden = encode(in_session,hidden)
#     for idx,sentence in enumerate(torch.transpose(in_session,1,0)):
#
#         encoder_optimizer.zero_grad()
#         decoder_optimizer.zero_grad()
#         context_optimizer.zero_grad()
#
#         context_input = sentence[-1]  #这里的context_input就可以代表一句话的全部信息
#         context_output,context_hidden=context(context_input,context_hidden)
#         decoder_hidden = context_hidden
#         decoder_input = torch.tensor([[SOS_token]], device=device)
#         use_teacher_forcing = True if random.random() < 0.5 else False
#         target_sentence = tar_session[idx]
#         target_length = len(target_sentence)
#         loss = 0
#
#         if use_teacher_forcing:
#             # Teacher forcing: Feed the target as the next input
#             for di in range(target_length):
#                 decoder_output,decode_hidden = decode(decoder_input,decoder_hidden)
#                 loss += criterion(decoder_output, Variable(torch.LongTensor([target_sentence[di]])))
#                 decoder_input = target_sentence[di]  # Teacher forcing
#         else:
#             # Without teacher forcing: use its own predictions as the next input
#             for di in range(target_length):
#                 decoder_output,decode_hidden = decode(decoder_input,decoder_hidden)
#                 topv, topi = decoder_output.data.topk(1)
#                 ni = topi[0][0]
#                 decoder_input = Variable(torch.LongTensor([[ni]]))
#                 loss += criterion(decoder_output, Variable(torch.LongTensor([target_sentence[di]])))
#                 if ni == EOS_token:
#                     break
#         loss.backward(retain_graph=True)
#         encoder_optimizer.step()
#         context_optimizer.step()
#         decoder_optimizer.step()
#     print(loss.item()/target_length)
