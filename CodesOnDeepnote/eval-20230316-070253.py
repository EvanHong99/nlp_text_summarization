# -*- coding: utf-8 -*-
# @File     : eval.py
# @Time     : 2022/3/4 13:02
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Project  : assignment1-NNLM
# @Description:
import torch
from torch import nn
from torch.autograd import Variable
import math
import time
import random
import os, sys

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
USE_CUDA=False

class FNN_LM(nn.Module):
    def __init__(self, nwords, emb_size, hid_size, num_hist,hid_size1=0):
        super(FNN_LM, self).__init__()
        self.embedding = nn.Embedding(nwords, emb_size)
        # 定义模型，包括层的类型以及参数个数
        if hid_size1!=0:
            self.fnn = nn.Sequential(
                nn.Linear(num_hist * emb_size, hid_size),
                nn.Tanh(),
                nn.Linear(hid_size,hid_size1),
                nn.Tanh(),
                nn.Linear(hid_size1, nwords)
            )
        else:
            self.fnn = nn.Sequential(
                nn.Linear(num_hist * emb_size, hid_size),
                nn.Tanh(),
                nn.Linear(hid_size, nwords)
            )
                

    def forward(self, words):
        emb = self.embedding(words)  # 3D Tensor of size [batch_size x num_hist x emb_size]
        feat = emb.view(emb.size(0), -1)  # 变形2D Tensor of size [batch_size x (num_hist*emb_size)]
        logit = self.fnn(feat)  # 2D Tensor of size [batch_size x nwords]

        return logit

w2i = {}
S = w2i["<s>"] = 0
UNK = w2i["<unk>"] = 1

# Functions to read in the corpus
# NOTE: We are using data from the Penn Treebank, which is already converted
#       into an easy-to-use format with "<unk>" symbols. If we were using other
#       data we would have to do pre-processing and consider how to choose
#       unknown words, etc.



def get_wid(w2i, x, add_vocab=True):
    """
    在word id 的dict中加入新的词，构成一个词典
    :param w2i:
    :param x:
    :param add_vocab:
    :return:
    """
    if x not in w2i:
        if add_vocab:
            w2i[x] = len(w2i)
        else:
            return UNK
    return w2i[x]


def read_dataset(filename, add_vocab):
    """

    :param filename:
    :param add_vocab:
    :return: iterator：不是很确定
    """
    display=True
    with open(filename, "r") as f:
        for line in f:
            if display:
                print(f"read dataset {[get_wid(w2i, x, add_vocab) for x in line.strip().split(' ')]}")
                display=False
            yield [get_wid(w2i, x, add_vocab) for x in line.strip().split(" ")]


# Read in the data
train = list(read_dataset("./data/ptb-text/train.txt", add_vocab=True))
dev = list(read_dataset("./data/ptb-text/valid.txt", add_vocab=False))
i2w = {v: k for k, v in w2i.items()}

nwords = len(w2i) # 语料库中的所有词数
N=3
EMB_SIZE = 128  # The size of the embedding
HID_SIZE = 128  # The size of the hidden layer
HID_SIZE1=0
sent=[]
for s in ["Jane went to the store","store to Jane went the"]:
    sent.append([get_wid(w2i, x, False) for x in s.strip().split(" ")])

PATH="./models/model_2022-03-07-15-26-44_tensor(396967.6875)/model_weight.pth"

# Initialize the model and the optimizer
model = FNN_LM(nwords=nwords, emb_size=EMB_SIZE, hid_size=HID_SIZE, num_hist=N,hid_size1=HID_SIZE1)
model.load_state_dict(torch.load(PATH))
model.eval()
print(model)

# convert a (nested) list of int into a pytorch Variable
def convert_to_variable(words):
    var = Variable(torch.LongTensor(words))
    if USE_CUDA:
        var = var.cuda()

    return var


# A function to calculate scores for one value
def calc_score_of_histories(words):
    # This will change from a list of histories, to a pytorch Variable whose data type is LongTensor
    words_var = convert_to_variable(words)
    logits = model(words_var)
    return logits


# Calculate the loss value for the entire sentence
def calc_sent_loss(sent):
    print("sent ",sent)
    # The initial history is equal to end of sentence symbols
    hist = [S] * N
    # Step through the sentence, including the end of sentence token
    all_histories = []
    all_targets = []
    for next_word in sent + [S]:
        all_histories.append(list(hist))
        all_targets.append(next_word)
        hist = hist[1:] + [next_word]
    print(all_histories)
    logits = calc_score_of_histories(all_histories)
    loss = nn.functional.cross_entropy(logits, convert_to_variable(all_targets), size_average=False)

    return loss

if __name__=="__main__":
    for s in sent:
        print(s)
        my_loss = calc_sent_loss(s) # sent是英文单词id list
        print("ppl(perplexity)=%.4f " % (math.exp(my_loss.data / len(s))))
        