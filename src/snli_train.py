import os
import sys
import math
from snli_preprocess import preprocess
from entailment_model import WACE
from batch import batcher_snli
from batch import evaluation
import torch
import torch.nn as nn


argvs = sys.argv
# parameters
gpu = 0
nli = argvs[1]
sequence_length = 30
input_size = 100
hidden_size = 100
num_layers = 1
emb_size = 100
num_classes = 3
learning_rate = 0.001
weight_decay = 0.001
dropout_rate = 0.2
num_epoch = 1000
train_batch = 124
dev_batch = 10000
test_batch = 10000

print("")
print("gpu:{0}".format(gpu))
print("nli:{0}".format(nli))
print("sequence_length:{0}".format(sequence_length))
print("input_size:{0}".format(input_size))
print("hidden_size:{0}".format(hidden_size))
print("num_layers:{0}".format(num_layers))
print("emb_size:{0}".format(emb_size))
print("num_classes:{0}".format(num_classes))
print("learning_rate:{0}".format(learning_rate))
print("weight_decay:{0}".format(weight_decay))
print("dropout_rate:{0}".format(dropout_rate))
print("num_epoch:{0}".format(num_epoch))
print("train_batch:{0}".format(train_batch))
print("dev_batch:{0}".format(dev_batch))
print("test_batch:{0}".format(test_batch))
print("")

dirpath = os.path.dirname(__file__)
glove_path = os.path.join(dirpath, "data/glove.6B.100d.h5")
data_path = os.path.join(dirpath, "../data")

preprocess = preprocess()
dataset = preprocess.load_dataset(data_path, nli)
preprocess.load_glove(glove_path)
token_data = preprocess.data_tokenization(dataset, sequence_length)
embedding = preprocess.embedding
vocab_size = len(preprocess.word2id)
print("vocab_size:{0}".format(vocab_size))
preprocess.save_index(data_path)
preprocess.save_embedding(data_path)

device = torch.device('cuda:{0}'.format(gpu) if torch.cuda.is_available() else 'cpu')

model = WACE(input_size, hidden_size, num_layers, num_classes, vocab_size, emb_size, embedding, dropout_rate, device).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, betas=(0.9, 0.999), weight_decay=weight_decay)

c_train = math.ceil(len(token_data["train"]["gold_label"])/train_batch)
c_dev = math.ceil(len(token_data["dev"]["gold_label"])/dev_batch)
c_test = math.ceil(len(token_data["test"]["gold_label"])/test_batch)
# train
print("Start Train")

for epoch in range(num_epoch):
    model.train()
    loss_sum = 0
    accuracy_sum = 0
    for premise, hypothesis, label in batcher_snli(token_data["train"], train_batch):
        x_p = torch.tensor(premise, dtype=torch.long, device=device)
        x_h = torch.tensor(hypothesis, dtype=torch.long, device=device)
        y = torch.tensor(label, dtype=torch.long, device=device)
        outputs = model(x_p, x_h)
        optimizer.zero_grad()
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        accuracy_sum += evaluation(outputs, y)
    print("Train | Epoch:{0} | Loss:{1} | Accuracy:{2}".format(epoch+1, loss_sum/c_train, accuracy_sum/c_train))
    if (epoch+1) % 1 == 0:
        model.eval()
        loss_sum = 0
        accuracy_sum = 0
        with torch.no_grad():
            for premise, hypothesis, label in batcher_snli(token_data["dev"], dev_batch):
                x_p = torch.tensor(premise, dtype=torch.long, device=device)
                x_h = torch.tensor(hypothesis, dtype=torch.long, device=device)
                y = torch.tensor(label, dtype=torch.long, device=device)
                outputs = model(x_p, x_h)
                loss = criterion(outputs, y)
                accuracy = evaluation(outputs, y)
                loss_sum += loss
                accuracy_sum += accuracy
        print("Dev | Epoch:{0} | Loss:{1} | Accuracy:{2}".format(epoch+1, loss_sum/c_dev, accuracy_sum/c_dev))
        torch.save(model.state_dict(), '../runs/{0}/parameters_{1}'.format(nli, epoch+1))
# test
model.eval()
loss_sum = 0
accuracy_sum = 0
with torch.no_grad():
    for premise, hypothesis, label in batcher_snli(token_data["test"], test_batch):
        x_p = torch.tensor(premise, dtype=torch.long, device=device)
        x_h = torch.tensor(hypothesis, dtype=torch.long, device=device)
        y = torch.tensor(label, dtype=torch.long, device=device)
        outputs = model(x_p, x_h)
        loss = criterion(outputs, y)
        accuracy = evaluation(outputs, y)
        loss_sum += loss
        accuracy_sum += accuracy
print("Test | Epoch:{0} | Loss:{1} | Accuracy:{2}".format(epoch+1, loss_sum/c_test, accuracy_sum/c_test))
# test
