#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import pickle as cPickle
import yaml
import codecs
import jieba
import numpy as np


def load_config():
    """
    Load hyper-parameters from the YAML file
    """
    fopen = open("config.yml")
    config = yaml.load(fopen, Loader=yaml.FullLoader)
    fopen.close()
    return config


class DataManager:
    """
    Manage the dataset, and the Input batch data
    """
    def __init__(self, batch_size=64, data_type='train', tags=[]):
        # Load some Hyper-parameters
        config = load_config()
        self.model_path = config.get("model_path")
        self.dataset_path = config.get("dataset_path")
        self.word_num = config.get("word_num")
        self.word_dim = config.get("word_dim")
        self.char_num = config.get("char_num")

        self.index = 0
        self.input_size = 0
        self.batch_size = batch_size
        self.data_type = data_type
        self.char_data = []
        self.word_data = []
        self.overall_data = []
        self.batch_data = []
        self.char_vocab = {"unk": 0}
        self.tag_map = {"O": 0, "START": 1, "STOP": 2}

        # If training the model, if there are no tags, arise errors
        if data_type == "train":
            # assert tags, Exception('What kinds of tags do you want to train?')
            self.generate_tags(tags)
            self.data_path = self.dataset_path + "train"
        elif data_type == "dev":
            self.data_path = self.dataset_path + "dev"
            self.load_char_map()
        elif data_type == "test":
            self.data_path = self.dataset_path + "test"
            self.load_char_map()

        # Load the Dataset and prepare batch
        global pre_trained
        pre_trained = self.load_word_vector()

        self.load_char_data()
        self.load_word_data()
        self.prepare_batch()

    def generate_tags(self, tags: list):
        """
        Generate tags for entities
        """
        self.tags = []
        for tag in tags:
            for prefix in ["B-", "I-", "E-", "S-"]:
                self.tags.append(prefix + tag)
        self.tags.append("O")

    def load_char_map(self):
        """
        Load data map from the "data.pkl" file
        """
        with open(self.model_path + "data.pkl", "rb") as f:
            self.data_map = cPickle.load(f)
            self.char_vocab = self.data_map.get("char_vocab", {})
            self.tag_map = self.data_map.get("tag_map", {})
            self.tags = self.data_map.keys()

    def load_char_data(self):
        """
        Load character data
        """
        sentence_index = []
        target_index = []
        self.char_data = []
        with open(self.data_path, encoding='UTF-8') as f:
            for line in f:
                line = line[:-1]
                # Segment sentences via "end " mark
                if line == "end ":
                    self.char_data.append([sentence_index, target_index])
                    sentence_index = []
                    target_index = []
                    continue

                character, tag = line.split(" ")

                # If character not in char_vocab --> Save the character
                if character not in self.char_vocab and self.data_type == "train":
                    self.char_vocab[character] = max(self.char_vocab.values()) + 1

                if tag not in self.tag_map and self.data_type == "train" and tag in self.tags:
                    self.tag_map[tag] = len(self.tag_map.keys())

                # Get the sentence and target tag
                sentence_index.append(self.char_vocab.get(character, 0))
                target_index.append(self.tag_map.get(tag, 0))

        self.input_size = len(self.char_vocab.values())
        return len(self.char_data)

    def load_word_vector(self):
        """
        Load word vectors
        """
        if 'pre_trained' not in globals().keys():
            print("Start to load pre-trained word embeddings!!")
            pre_trained = {}
            for i, line in enumerate(codecs.open(self.model_path + "word_vectors.vec", 'r', encoding='utf-8')):
                line = line.rstrip().split()
                if len(line) == self.word_dim + 1:
                    pre_trained[line[0]] = np.array([float(x) for x in line[1:]]).astype(np.float32)
        else:
            pre_trained = globals().get("pre_trained")
        return pre_trained

    def load_word_data(self):
        """
        Load word data
        """
        # Load word vectors
        pre_trained = self.load_word_vector()

        sentence = []
        embed_words = []
        with open(self.data_path, encoding='UTF-8') as f:
            for line in f:
                character = line[:-1][0]
                if line[:-1][0:3] == "end":
                    sentence = ''.join(sentence)
                    words = jieba.lcut(sentence, HMM=True)

                    for i in words:
                        vec = pre_trained.get(i)
                        if str(type(vec)) != "<class 'NoneType'>":
                            embed_words.append(vec)
                        else:
                            gen_vec = np.random.normal(size=self.word_dim).tolist()
                            embed_words.append(gen_vec)

                    self.word_data.append(embed_words)
                    sentence = []
                    embed_words = []
                    continue
                sentence.append(character)

    # characters, tags, words, words length, characters length
    def prepare_batch(self):
        """
        prepare batch
        """
        self.chars = np.array(self.char_data, dtype=object)
        self.words, self.word_length = self.pad_word_data(self.word_data)
        self.words = np.array(self.words, dtype=object)
        self.words = np.expand_dims(self.words, axis=1)
        self.word_length = np.array(self.word_length, dtype=object)
        self.word_length = np.expand_dims(self.word_length, axis=1)
        self.overall_data = np.concatenate([self.chars, self.words, self.word_length], axis=1).tolist()

        for i in range(len(self.overall_data)):
            words = self.overall_data[i][2]
            self.overall_data[i][2] = words[:self.word_num*self.word_dim]

            word_length = self.overall_data[i][3]
            if word_length > self.word_num:
                self.overall_data[i][3] = self.word_num

        index = 0
        while True:
            if index + self.batch_size >= len(self.overall_data):
                pad_data = self.pad_char_data(self.overall_data[-self.batch_size:])
                self.batch_data.append(pad_data)
                break
            else:
                pad_data = self.pad_char_data(self.overall_data[index:index+self.batch_size])
                index += self.batch_size
                self.batch_data.append(pad_data)

    def pad_char_data(self, data: list):
        c_data = copy.deepcopy(data)
        for i in c_data:
            if len(i[0]) >= self.char_num:
                i.append(self.char_num)
                i[0] = i[0][:self.char_num]
                i[1] = i[1][:self.char_num]
            else:
                i.append(len(i[0]))
                i[0] = i[0] + (self.char_num - len(i[0])) * [0]
                i[1] = i[1] + (self.char_num - len(i[1])) * [0]
        return c_data

    def pad_word_data(self, data: list):
        output_data = []
        word_length = []
        c_data = copy.deepcopy(data)
        for i in range(len(c_data)):
            word_length.append(len(c_data[i]))
            if len(c_data[i]) <= self.word_num:
                c_data[i] = c_data[i] + (self.word_num - len(c_data[i])) * [[0] * self.word_dim]
            c_data[i] = np.reshape(c_data[i], [np.shape(c_data[i])[0] * np.shape(c_data[i])[1]])
            c_data[i] = np.concatenate([c_data[i], np.random.randint(0, 5) * [0]], axis=0)
            output_data.append(c_data[i])
        return output_data, word_length

    def iteration(self):
        """
        Batch iteration
        """
        idx = 0
        while True:
            yield self.batch_data[idx]
            idx += 1
            if idx > len(self.batch_data) - 1:
                idx = 0

    def get_batch(self):
        """
        Get batch
        """
        for data in self.batch_data:
            yield data
