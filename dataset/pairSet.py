from torch.utils.data import Dataset
from Config import *
import torch

class pairSet(Dataset):
    def __init__(self,pairs,lang):
        self.pairs = pairs
        self.lang = lang

    def __getitem__(self, index):
        return self.tensorsFromPair(self.pairs[index])

    def __len__(self):
        return len(self.pairs)

    def get_sentence(self,index):
        return self.pairs[index][0], self.pairs[index][1]

    def indexesFromSentence(self, sentence):
        return [self.lang.word2index[word] for word in sentence.split(' ')]

    def tensorFromSentence(self, sentence):
        indexes = self.indexesFromSentence( sentence)
        indexes.append(EOS_token)
        return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

    def tensorsFromPair(self,pair):
        input_tensor = self.tensorFromSentence(pair[0])
        target_tensor = self.tensorFromSentence(pair[1])
        return (input_tensor, target_tensor)




