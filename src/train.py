import os
import sys
import math
from preprocess import pre_CE
from entailment_representation import entailment_representation
from model import NeuralNet
from model import LSTM
from model import CE
from model import ACE
from model import WACE
from batch import batcher
from batch import evaluation
from batch import report_F1
from score import report_score
import torch
import torch.nn as nn

# model select
argvs = sys.argv
argc = len(argvs)
if argc < 5:
    print("moedl select:[NN, LSTM, CE, ACE, WACE]")
    print("feature select:[cossim, entailment]")
    print("gpu select:[int]")
    print("sequence_length select:[int]")
preprocess_dict = {'NN': pre_CE(), 'LSTM': pre_CE(), 'CE': pre_CE(), 'ACE': pre_CE(), 'WACE': pre_CE()}
# parameters
gpu = argvs[3]
head_length = 30
body_length = int(argvs[4])
top_word = 5000
input_size = 100
hidden_size = 100
num_layers = 1
emb_size = 100
num_classes = 4
learning_rate = 0.001
weight_decay = 0.001
dropout_rate = 0.2
num_epoch = 100
train_batch = 124
dev_batch = 100
test_batch = 100
feature = "None"
cosine_similarity = False
entailment = False
if argvs[2] == 'cossim':
    feature = "cosine_similarity"
    cosine_similarity = True
if argvs[2] == 'entailment':
    feature = "entailment"
    entailment = True

print("")
print("model:{0}".format(argvs[1]))
print("feature:{0}".format(feature))
print("gpu:{0}".format(gpu))
print("head_length:{0}".format(head_length))
print("body_length:{0}".format(body_length))
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
glove_path = os.path.join(dirpath, "../data/glove.6B.100d.h5")
w2v_path = os.path.join(dirpath, "../data/GoogleNews-vectors-negative300.bin")
data_path = os.path.join(dirpath, "../data")

preprocess = preprocess_dict[argvs[1]]
dataset = preprocess.load_dataset(data_path)
if argvs[1] != 'NN':
    # preprocess.load_w2v(w2v_path)
    preprocess.load_glove(glove_path)
    id2word_snli, word2id_snli = preprocess.load_index(data_path)
    embedding_snli = preprocess.load_embedding(data_path)
    preprocess.init_word_snli(id2word_snli, word2id_snli)
    token_data, sent_data = preprocess.data_tokenization(dataset, head_length, body_length)
    embedding = preprocess.embedding
    tfidf_cossim = preprocess.tfidf(dataset, top_word)
    entailment_rep = entailment_representation(sent_data, word2id_snli, embedding_snli)
else:
    token_data, sent_data = preprocess.data_tokenization(dataset, head_length, body_length)
    tfidf_cossim = preprocess.tfidf(dataset, top_word)
'''
for en, head, body, d in zip(entailment_rep["train"], sent_data["train"]["Headline"], sent_data["train"]["articleBody"], dataset["train"]["Stance"]):
    for e, b in zip(en, body):
        print(d)
        print([id2word_snli[word_id] for word_id in head])
        print([id2word_snli[word_id] for word_id in b])
        print(e)
'''
# embedding = preprocess.embedding
vocab_size = len(preprocess.word2id)
print("vocab_size:{0}".format(vocab_size))

device = torch.device('cuda:{0}'.format(gpu) if torch.cuda.is_available() else 'cpu')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if argvs[1] == 'NN':
    model = NeuralNet(input_size, hidden_size, num_layers, num_classes, vocab_size, emb_size, dropout_rate, cosine_similarity, device).to(device)
if argvs[1] == 'LSTM':
    model = LSTM(input_size, hidden_size, num_layers, num_classes, vocab_size, emb_size, embedding, dropout_rate, cosine_similarity, device).to(device)
if argvs[1] == 'CE':
    model = CE(input_size, hidden_size, num_layers, num_classes, vocab_size, emb_size, embedding, dropout_rate, cosine_similarity, device).to(device)
if argvs[1] == 'ACE':
    model = ACE(input_size, hidden_size, num_layers, num_classes, vocab_size, emb_size, embedding, dropout_rate, cosine_similarity, device).to(device)
if argvs[1] == 'WACE':
    model = WACE(input_size, hidden_size, num_layers, num_classes, vocab_size, emb_size, embedding, dropout_rate, cosine_similarity, entailment, device).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, betas=(0.9, 0.999), weight_decay=weight_decay)

c_train = math.ceil(len(token_data["train"]["Stance"])/train_batch)
# c_dev = math.ceil(len(token_data["dev"]["Stance"])/dev_batch)
c_test = math.ceil(len(token_data["test"]["Stance"])/test_batch)
# train
print("Start Train")
for epoch in range(num_epoch):
    model.train()
    loss_sum = 0
    accuracy_sum = 0
    for premise, hypothesis, label, sim, entail in batcher(token_data["train"], tfidf_cossim["train"], entailment_rep["train"], train_batch):
        x_p = torch.tensor(premise, dtype=torch.long, device=device)
        x_h = torch.tensor(hypothesis, dtype=torch.long, device=device)
        y = torch.tensor(label, dtype=torch.long, device=device)
        x_similarity = torch.tensor(sim, dtype=torch.float, device=device)
        x_entailment = torch.stack(entail, dim=0).to(device)
        outputs = model(x_p, x_h, x_similarity, x_entailment)
        optimizer.zero_grad()
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        accuracy_sum += evaluation(outputs, y)
    print("Train | Epoch:{0} | Loss:{1} | Accuracy:{2}".format(epoch+1, loss_sum/c_train, accuracy_sum/c_train))
    if (epoch+1) % 1 == 0:
        out = []
        y_label = []
        model.eval()
        with torch.no_grad():
            for premise, hypothesis, label, sim, entail in batcher(token_data["test"], tfidf_cossim["test"], entailment_rep["test"], test_batch):
                x_p = torch.tensor(premise, dtype=torch.long, device=device)
                x_h = torch.tensor(hypothesis, dtype=torch.long, device=device)
                y = torch.tensor(label, dtype=torch.long, device=device)
                x_similarity = torch.tensor(sim, dtype=torch.float, device=device)
                x_entailment = torch.stack(entail, dim=0).to(device)
                outputs = model(x_p, x_h, x_similarity, x_entailment)
                out.append(outputs)
                y_label.append(y)
            outputs = torch.cat(out, dim=0)
            y = torch.cat(y_label)
            loss = criterion(outputs, y)
            accuracy = evaluation(outputs, y)
        print("Dev | Epoch:{0} | Loss:{1} | Accuracy:{2}".format(epoch+1, loss, accuracy))
        report_F1(outputs, y)
        report_score(outputs, y)
        # torch.save(model.state_dict(), 'runs/{0}/parameters_{1}'.format(argvs[1], epoch+1))
'''
# test
model.eval()
loss_sum = 0
accuracy_sum = 0
with torch.no_grad():
    for premise, hypothesis, label in batcher(token_data["test"], test_batch):
        x_p = torch.tensor(premise, dtype=torch.long, device=device)
        x_h = torch.tensor(hypothesis, dtype=torch.long, device=device)
        y = torch.tensor(label, dtype=torch.long, device=device)
        outputs = model(x_p, x_h)
        loss = criterion(outputs, y)
        accuracy = evaluation(outputs, y)
        loss_sum += loss
        accuracy_sum += accuracy
print("Test | Epoch:{0} | Loss:{1} | Accuracy:{2}".format(epoch+1, loss_sum/c_test, accuracy_sum/c_test))
'''
