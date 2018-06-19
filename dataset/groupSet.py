from torch.utils.data import Dataset
from Config import *
import torch
import numpy as np

class groupSet(Dataset):

    def __init__(self,lang,batch_size = 30):

        self.lang = lang
        self.sessions = None
        self.batch_size = batch_size

    def __getitem__(self, index):
        return self.tensorsFromSession(self.sessions[index])

    def __len__(self):
        return len(self.sessions)

    def loadfnp(self,path):
        self.sessions = np.load(path)

    def indexesFromSentence(self, sentence):
        return [self.lang.word2index[word] for word in sentence.split(' ')]

    def tensorFromSentence(self, sentence):
        indexes = self.indexesFromSentence( sentence)
        indexes.append(EOS_token)
        return indexes,len(indexes)

    def tensorsFromSession(self,session):
        temp = []
        max = -1
        for sentence in session:
            tmp,length = self.tensorFromSentence(sentence)
            temp.append(tmp)
            if(length > max):
                max = length
        return self.pad2batch(temp,max),len(session)

    def pad2batch(self,temp,max):
        ans = torch.zeros(self.batch_size,max,dtype=torch.long,device = device)
        for idx ,item in enumerate(temp):
            if(idx>=self.batch_size):
                break
            ans[idx][0:len(item)] = torch.from_numpy(np.array(item))
        return ans

