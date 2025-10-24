# data.py
import unicodedata
import re
import random
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import numpy as np

# 添加device定义
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ", "he is", "he s ", "she is", "she s ",
    "you are", "you re ", "we are", "we re ", "they are", "they re "
)

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {SOS_token: "SOS", EOS_token: "EOS"}
        self.n_words = 2  # Count SOS and EOS

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

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s.strip()

def filterPair(p):
    return len(p[0].split(" ")) < MAX_LENGTH and \
           len(p[1].split(" ")) < MAX_LENGTH and \
           p[1].startswith(eng_prefixes)

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def read_pairs_from_file(path, reverse=False):
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    pairs = []
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) >= 4:
            src = normalizeString(parts[1])  # English
            tgt = normalizeString(parts[3])  # French
            pair = [tgt, src] if reverse else [src, tgt]
            pairs.append(pair)
        if len(parts) == 2:
            pair = [normalizeString(s) for s in parts]
            pairs.append(pair)
    return pairs

def prepareData(path, reverse=False):
    pairs = read_pairs_from_file(path, reverse)
    print("Read %d sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %d sentence pairs with prefix filtering and max length" % len(pairs))

    input_lang = Lang("fra" if reverse else "eng")
    output_lang = Lang("eng" if reverse else "fra")

    print("Building vocabularies...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Input Lang size:", input_lang.n_words)
    print("Output Lang size:", output_lang.n_words)
    return input_lang, output_lang, pairs

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ') if word in lang.word2index]

def get_dataloader(path, batch_size=32, reverse=True):
    input_lang, output_lang, pairs = prepareData(path, reverse=reverse)

    n = len(pairs)
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)

    for i, (inp, tgt) in enumerate(pairs):
        inp_idx = indexesFromSentence(input_lang, inp)
        tgt_idx = indexesFromSentence(output_lang, tgt)
        inp_idx.append(EOS_token)
        tgt_idx.append(EOS_token)

        inp_idx = inp_idx[:MAX_LENGTH]
        tgt_idx = tgt_idx[:MAX_LENGTH]

        input_ids[i, :len(inp_idx)] = inp_idx
        target_ids[i, :len(tgt_idx)] = tgt_idx

    input_tensor = torch.LongTensor(input_ids).to(device)
    target_tensor = torch.LongTensor(target_ids).to(device)
    dataset = TensorDataset(input_tensor, target_tensor)

    dataloader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=batch_size)
    return input_lang, output_lang, dataloader

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long).view(-1, 1).to(device)

