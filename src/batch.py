import numpy as np
import torch
from sklearn.metrics import classification_report


def batcher(dataset, tfidf_cossim, entailment_rep, num_batch):
    N = len(dataset["Stance"])
    ids = torch.randperm(N)
    for i in range(0, N, num_batch):
        x_p_batch = []
        x_h_batch = []
        y_batch = []
        similarity = []
        entailment = []
        for num in ids[i:i+num_batch]:
            x_p_batch.append(dataset["Headline"][num])
            x_h_batch.append(dataset["articleBody"][num])
            y_batch.append(dataset["Stance"][num])
            similarity.append(tfidf_cossim[num])
            entailment.append(entailment_rep[num])
        yield x_p_batch, x_h_batch, y_batch, similarity, entailment


def batcher_entailment(dataset):
    N = len(dataset["Headline"])
    for i in range(N):
        x_p_batch = []
        x_h_batch = []
        for j in range(len(dataset["articleBody"][i])):
            x_h_batch.append(dataset["Headline"][i])
            x_p_batch.append(dataset["articleBody"][i][j])
        yield x_p_batch, x_h_batch


def batcher_snli(dataset, num_batch):
    N = len(dataset["gold_label"])
    ids = torch.randperm(N)
    for i in range(0, N, num_batch):
        x_p_batch = []
        x_h_batch = []
        y_batch = []
        for num in ids[i:i+num_batch]:
            x_p_batch.append(dataset["sentence1"][num])
            x_h_batch.append(dataset["sentence2"][num])
            y_batch.append(dataset["gold_label"][num])
        yield x_p_batch, x_h_batch, y_batch

def evaluation(prediction, label):
    prediction = prediction.cpu().detach().numpy()
    label = label.cpu().detach().numpy()
    prediction = np.argmax(prediction, axis=1)
    accuracy = np.sum(prediction == label) / len(label)
    return accuracy


def report_F1(prediction, label):
    prediction = prediction.cpu().detach().numpy()
    label = label.cpu().detach().numpy()
    prediction = np.argmax(prediction, axis=1)
    print(classification_report(label, prediction, digits=3))
