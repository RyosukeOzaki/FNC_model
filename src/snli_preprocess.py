import os
import sys
import pickle
import pandas as pd
import numpy as np
import gensim
import h5py


class preprocess():
    def __init__(self):
        PAD = '<PAD>'
        UNK = '<UNK>'
        DELIMITER = '<DELIMITER>'
        pad_emb = np.zeros((1, 100))
        unk_emb = np.random.randn(1, 100)
        delimiter_emb = np.random.randn(1, 100)
        self.word2id = {PAD: 0, UNK: 1, DELIMITER: 2}
        self.id2word = {0: PAD, 1: UNK, 2: DELIMITER}
        self.embedding = np.append(pad_emb, np.append(unk_emb, delimiter_emb, axis=0), axis=0)
        self.word2vec = None

    def load_dataset(self, fname, file):
        print("Loading Data")
        dataset = {}
        if file == "snli":
            for type_set in ["train", "dev", "test"]:
                dataset[type_set] = pd.read_csv(fname+"/snli_1.0_{0}.csv".format(type_set), usecols=["sentence1", "sentence2", "gold_label"])
        else:
            for type_set in ["train", "dev", "test"]:
                dataset[type_set] = pd.read_csv(fname+"/multinli_1.0_{0}.txt".format(type_set), sep="\t", usecols=["sentence1", "sentence2", "gold_label"])
        print("Finish\n")
        return dataset

    def load_glove(self, glove_path):
        print('Loading Glove')
        glove = h5py.File(glove_path, 'r')
        words = glove['key'].value
        self.embedding = np.append(self.embedding, glove['value'].value, axis=0)
        for word in words:
                if not word in self.word2id:
                    self.word2id[word] = len(self.word2id)
                    self.id2word[len(self.id2word)] = word
        print('Finish\n')


    def load_w2v(self, w2v_path):
        print("Loading Word2vec")
        self.word2vec = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=True)
        print('Finish\n')

    def clean_sequence_to_words(self, sequence):
        sequence = sequence.lower()
        punctuations = [".", ",", ";", "!", "?", "/", '"', "'", "(", ")", "{", "}", "[", "]", "="]
        for punctuation in punctuations:
            sequence = sequence.replace(punctuation, " {} ".format(punctuation))
        sequence = sequence.replace("  ", " ")
        sequence = sequence.replace("   ", " ")
        sequence = sequence.split(" ")
        todelete = ["", " ", "  "]
        for i, elt in enumerate(sequence):
            if elt in todelete:
                sequence.pop(i)
        return sequence

    def data_id(self, token, train=False):
        # if train:
            # for word in token:
                # if not word in self.word2id:
                    # self.word2id[word] = len(self.word2id)
                    # self.id2word[len(self.id2word)] = word
                    # self.embedding = np.append(self.embedding, np.random.uniform(-0.05, 0.05, (1, 100)), axis=0)
        data_id = [self.word2id[word] if word in self.word2id else self.word2id['UNKNOWN'] for word in token]
        return data_id

    def padding(self, data, sequence_length, hypothesis=False):
        if len(data) >= sequence_length - int(hypothesis):
            n_data = data[:sequence_length - int(hypothesis)]
        else:
            padding = [self.word2id['<PAD>'] for i in range(sequence_length - len(data) - int(hypothesis))]
            n_data = data + padding
        if hypothesis:
            n_data = [self.word2id['<DELIMITER>']] + n_data
        return n_data

    def data_tokenization(self, dataset, sequence_length):
        print("Tokenization")
        token_data = dict((type_set, {"sentence1": [], "sentence2": [], "gold_label": []}) for type_set in ["train", "dev", "test"])
        map_label = {"entailment": 0, "neutral": 1, "contradiction": 2}
        for type_set in ["train", "dev", "test"]:
            id = len(dataset[type_set]["gold_label"])
            for num in range(id):
                try:
                    token_sent1 = self.padding(self.data_id(self.clean_sequence_to_words(dataset[type_set]["sentence1"][num]), train=type_set == 'train'), sequence_length)
                    token_sent2 = self.padding(self.data_id(self.clean_sequence_to_words(dataset[type_set]["sentence2"][num]), train=type_set == 'train'), sequence_length, hypothesis=True)
                    token_gold = map_label[dataset[type_set]["gold_label"][num]]
                except:
                    pass
                else:
                    token_data[type_set]["sentence1"].append(token_sent1)
                    token_data[type_set]["sentence2"].append(token_sent2)
                    token_data[type_set]["gold_label"].append(token_gold)
                sys.stdout.write("\r{}_id: {}/{}   ".format(type_set, num+1, id))
                sys.stdout.flush()
            print("")
        print("Finish tokenization\n")
        return token_data

    def init_self(self, word2id, id2word, embedding):
        self.id2word = id2word
        self.word2id = word2id
        self.embedding = embedding

    def save_index(self, dirpath):
        pickle.dump((self.id2word, self.word2id), open(dirpath+"/dicts.pickle", 'wb'))

    def load_index(self, dirpath):
        id2word, word2id = pickle.load(open(dirpath+"/dicts.pickle", 'rb'))
        return id2word, word2id

    def save_embedding(self, dirpath):
        pickle.dump((self.embedding), open(dirpath+"/embedding.pickle", 'wb'))

    def load_embedding(self, dirpath):
        embedding = pickle.load(open(dirpath+"/embedding.pickle", 'rb'))
        return embedding
