from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


class CharDataset(Dataset):

    def __init__(self, words, chars, max_word_length):
        self.words = words
        self.chars = chars
        self.max_word_length = max_word_length
        self.stoi = {ch:i+1 for i,ch in enumerate(chars)}
        self.itos = {i:s for s,i in self.stoi.items()} 

    def __len__(self):
        return len(self.words)

    def contains(self, word):
        return word in self.words

    def get_vocab_size(self):
        return len(self.chars) + 1

    def get_output_length(self):
        return self.max_word_length + 1 

    def encode(self, word):
        ix = torch.tensor([self.stoi[w] for w in word], dtype=torch.long) 
        return ix

    def decode(self, ix):
        word = ''.join(self.itos[i] for i in ix) 
        return word

    def __getitem__(self, idx):
        word = self.words[idx]
        ix = self.encode(word)
        x = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        y = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        x[1:1+len(ix)] = ix
        y[:len(ix)] = ix
        y[len(ix)+1:] = -1 
        return x, y

def create_datasets(input_file):

    with open(input_file, 'r') as f:
        data = f.read()
    words = data.splitlines()
    words = [w.strip() for w in words]
    words = [w for w in words if w]
    chars = ['+', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '='] 
    max_word_length = max(len(w) for w in words)

    dataset = CharDataset(words, chars, max_word_length)
    return dataset
