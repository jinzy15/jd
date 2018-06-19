import utils.Normalizer as Norm
from Config import *
import numpy as np
import pickle as pkl

class Lang:
    def __init__(self, name,file):
        self.name = name
        self.file = file
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "<pad>",1: "SOS", 2: "EOS"}
        self.n_words = 3  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def readLangs(self,reverse=False):
        print("Reading lines...")
        # Read the file and split into lines
        lines = open(self.file, encoding='utf-8').read().strip().split('\n')
        # Split every line into pairs and normalize
        pairs = [[Norm.ch_normalizeString(l.split('\t')[0]),
                  Norm.ch_normalizeString(l.split('\t')[1])] for l in lines]
        # Reverse pairs, make Lang instances
        if reverse:
            pairs = [list(reversed(p)) for p in pairs]
        return pairs

    def filterPair(self,p):
        return len(p[0].split(' ')) < MAX_LENGTH and \
               len(p[1].split(' ')) < MAX_LENGTH

    def filterPairs(self,pairs):
        return [pair for pair in pairs if self.filterPair(pair)]

    def prepareData(self,reverse=False):
        pairs = self.readLangs(reverse)
        print("Read %s sentence pairs" % len(pairs))
        pairs = self.filterPairs(pairs)
        print("Trimmed to %s sentence pairs" % len(pairs))
        print("Counting words...")
        for pair in pairs:
            self.addSentence(pair[0])
            self.addSentence(pair[1])
        print("Counted words:")
        print(self.name, self.n_words)
        return pairs

class npLang(Lang):
    def prepareData(self):
        sessions = np.load(self.file)
        for session in sessions:
            for sentence in session:
                self.addSentence(sentence)
        print("Counted words:")
        print(self.name, self.n_words)

    def saveLang(self,name):
        pkl.dump([self.word2index,
        self.word2count,
        self.index2word,
        self.n_words],open(name,'wb'))

    def loadLang(self,name):
        [self.word2index,
         self.word2count,
         self.index2word,
         self.n_words] = pkl.load(open(name,'rb'))