import os
import sys
import pickle
import pandas as pd
import numpy as np
import gensim
import h5py
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class pre_CE():
    def __init__(self):
        PAD = '<PAD>'
        UNK = '<UNK>'
        DELIMITER = '<DELIMITER>'
        pad_emb = np.zeros((1, 100))
        unk_emb = np.random.randn(1, 100)
        delimiter_emb = np.random.randn(1, 100)
        self.nlp = spacy.load('en')
        self.word2id = {PAD: 0, UNK: 1, DELIMITER: 2}
        self.id2word = {0: PAD, 1: UNK, 2: DELIMITER}
        self.embedding = np.append(pad_emb, np.append(unk_emb, delimiter_emb, axis=0), axis=0)
        self.word2id_snli = None
        self.id2word_snli = None
        self.word2vec = None

    def load_dataset(self, fname):
        print("Loading Data")
        dataset = {}
        for type_set in ["train", "test"]:
            dataset[type_set] = pd.read_csv(fname+"/{0}_FNC.csv".format(type_set), usecols=["Headline", "articleBody", "Stance"])
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
        punctuations = [".", ",", ";", "!", "?", "/", '"', "'", "(", ")", "{", "}", "[", "]", "="]
        for punctuation in punctuations:
            sequence = sequence.replace(punctuation, " {} ".format(punctuation))
        sequence = sequence.replace("  ", " ")
        sequence = sequence.replace("   ", " ")
        doc = self.nlp(sequence)
        sequence_split = []
        for token in doc:
            if token.lemma_ == "-PRON-":
                sequence_split.append(token.lower_)
            else:
                sequence_split.append(token.lemma_)
        todelete = ["", " ", "  "]
        for i, elt in enumerate(sequence_split):
            if elt in todelete:
                sequence_split.pop(i)
        return sequence_split

    def padding_sent(self, sentences, length=25):
        if len(sentences) >= length:
            n_sentences = sentences[:length]
        else:
            padding = [["<PAD>"] for i in range(length - len(sentences))]
            n_sentences = sentences + padding
        return n_sentences

    def split_sequence(self, sequence):
        sentences = []
        doc = self.nlp(sequence)
        for sent in doc.sents:
            sentences.append(self.clean_sequence_to_words(sent.text))
        sentences = self.padding_sent(sentences)
        return sentences

    def data_id(self, token, train=False):
        if train:
            for word in token:
                if not word in self.word2id:
                    self.word2id[word] = len(self.word2id)
                    self.id2word[len(self.id2word)] = word
                    self.embedding = np.append(self.embedding, np.random.uniform(-0.05, 0.05, (1, 100)), axis=0)
        data_id = [self.word2id[word] if word in self.word2id else self.word2id['UNKNOWN'] for word in token]
        return data_id

    def data_id_snli(self, token):
        data_id = [self.word2id_snli[word] if word in self.word2id_snli else self.word2id_snli['UNKNOWN'] for word in token]
        return data_id

    def padding(self, data, sequence_length, body=False):
        if len(data) >= sequence_length - int(body):
            n_data = data[:sequence_length - int(body)]
        else:
            padding = ['<PAD>' for i in range(sequence_length - len(data) - int(body))]
            n_data = data + padding
        if body:
            n_data = ['<DELIMITER>'] + n_data
        return n_data

    def data_tokenization(self, dataset, head_length, body_length):
        print("Tokenization")
        token_data = dict((type_set, {"Headline": [], "articleBody": [], "Stance": []}) for type_set in ["train", "test"])
        sent_data = dict((type_set, {"Headline": [], "articleBody": []}) for type_set in ["train", "test"])
        map_label = {"agree": 0, "disagree": 1, "discuss": 2, "unrelated": 3}
        for type_set in ["train", "test"]:
            id = len(dataset[type_set]["Stance"])
            for num in range(id):
                clean_head = self.clean_sequence_to_words(dataset[type_set]["Headline"][num])
                clean_body = self.clean_sequence_to_words(dataset[type_set]["articleBody"][num])
                padding_head = self.padding(clean_head, head_length)
                padding_body = self.padding(clean_body, body_length, body=True)
                token_head = self.data_id(padding_head, train=type_set == 'train')
                token_body = self.data_id(padding_body, train=type_set == 'train')
                token_stance = map_label[dataset[type_set]["Stance"][num]]
                sent_data[type_set]["Headline"].append(self.data_id_snli(padding_head))
                sent_data[type_set]["articleBody"].append([self.data_id_snli(self.padding(sentence, 50)) for sentence in self.split_sequence(dataset[type_set]["articleBody"][num])])
                token_data[type_set]["Headline"].append(token_head)
                token_data[type_set]["articleBody"].append(token_body)
                token_data[type_set]["Stance"].append(token_stance)
                sys.stdout.write("\r{}_id: {}/{}   ".format(type_set, num+1, id))
                sys.stdout.flush()
            print("")
        print("Finish\n")
        return token_data, sent_data

    def tfidf(self, dataset, top_word):
        print("TFIDF Vectorization")
        tfidf_vec = TfidfVectorizer(max_features=top_word).fit(list(map(str, dataset["train"]["Headline"])) + list(map(str, dataset["train"]["articleBody"])) + list(map(str, dataset["test"]["Headline"])) + list(map(str, dataset["test"]["articleBody"])))
        tfidf_cossim = {"train": [], "test": []}
        for type_set in ["train", "test"]:
            id = len(dataset[type_set]["Stance"])
            for num in range(id):
                tfidf_head = tfidf_vec.transform([dataset[type_set]["Headline"][num]]).toarray()
                tfidf_body = tfidf_vec.transform([dataset[type_set]["articleBody"][num]]).toarray()
                tfidf_cossim[type_set].append(cosine_similarity(tfidf_head, tfidf_body)[0])
                sys.stdout.write("\r{}_id: {}/{}   ".format(type_set, num+1, id))
                sys.stdout.flush()
            print("")
        print("Finish\n")
        return tfidf_cossim

    def init_word_snli(self, id2word, word2id):
        self.id2word_snli = id2word
        self.word2id_snli = word2id

    def load_index(self, dirpath):
        id2word, word2id = pickle.load(open(dirpath+"/dicts.pickle", 'rb'))
        return id2word, word2id

    def load_embedding(self, dirpath):
        embedding = pickle.load(open(dirpath+"/embedding.pickle", 'rb'))
        return embedding

if __name__ == "__main__":
    dirpath = os.path.dirname(__file__)
    glove_path = os.path.join(dirpath, "data/glove.6B.100d.h5")
    w2v_path = os.path.join(dirpath, "../data/GoogleNews-vectors-negative300.bin")
    data_path = os.path.join(dirpath, "data")
    p = pre_CE()
    dataset = p.load_dataset(data_path)
    # p.load_glove(glove_path)
    # token_data = p.data_tokenization(dataset, 50)
