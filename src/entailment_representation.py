import os
import sys
from entailment_model import WACE
from batch import batcher_entailment
import torch
import torch.nn as nn


def entailment_representation(sent_data, word2id, embedding):
    # parameters
    gpu = 0
    input_size = 100
    hidden_size = 100
    num_layers = 1
    emb_size = 100
    num_classes = 3
    dropout_rate = 0.2

    # embedding = preprocess.embedding
    vocab_size = len(word2id)
    embedding = embedding
    device = torch.device('cuda:{0}'.format(gpu) if torch.cuda.is_available() else 'cpu')
    model = WACE(input_size, hidden_size, num_layers, num_classes, vocab_size, emb_size, embedding, dropout_rate, device).to(device)

    nli = input("(snli:0/multi:1)")
    num = input("num:")
    if nli == "0":
        nli = "snli"
    else:
        nli = "multinli"
    model.load_state_dict(torch.load("../runs/{0}/parameters_{1}".format(nli, num)))
    model.eval()
    entailment_rep = {"train": [], "test": []}
    with torch.no_grad():
        for type_set in ["train", "test"]:
            for premise, hypothesis in batcher_entailment(sent_data[type_set]):
                x_p = torch.tensor(premise, dtype=torch.long, device=device)
                x_h = torch.tensor(hypothesis, dtype=torch.long, device=device)
                outputs = model(x_p, x_h)
                entailment_rep[type_set].append(outputs)
    return entailment_rep
