#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import necessary Python Packages
import copy
import pickle as cPickle
import yaml
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
    def __init__(self, batch_size=64, data_type='train', tags=None):
        if tags is None:
            tags = []

        config = load_config()
        self.dataset_path = config.get("dataset_path")
        self.model_path = config.get("model_path")
        self.max_length = config.get("max_length")

        self.index = 0
        self.input_size = 0
        self.batch_size = batch_size
        self.data_type = data_type
        self.data = []
        self.batch_data = []
        self.vocab = {"unk": 0}
        self.tag_map = {"O": 0, "START": 1, "STOP": 2}

        # If training the model, if there are no tags, arise errors
        if data_type == "train":
            self.generate_tags(tags)
            self.data_path = self.dataset_path + "train"

        elif data_type == "dev":
            self.data_path = self.dataset_path + "dev"
            self.load_data_map()

        # Load the Dataset and prepare batch
        self.load_data()
        self.prepare_batch()

    def generate_tags(self, tags: list):
        """
        If there are no tags, generate tags (for Exception)
        :param tags: tags list containing all possible tags
        """
        self.tags = []
        for tag in tags:
            for prefix in ["B-", "I-", "E-", "S-"]:
                self.tags.append(prefix + tag)
        self.tags.append("O")

    def load_data_map(self):
        with open(self.model_path + "data.pkl", "rb") as f:
            self.data_map = cPickle.load(f)
            self.vocab = self.data_map.get("vocab", {})
            self.tag_map = self.data_map.get("tag_map", {})
            self.tags = self.data_map.keys()

    def load_data(self):
        """
        load data and add vocab
        """
        sentence = []
        target = []
        with open(self.data_path, encoding='UTF-8') as f:
            for line in f:
                line = line[:-1]

                # Segment sentences via "end " mark
                if line == "end ":
                    self.data.append([sentence, target])
                    sentence = []
                    target = []
                    continue

                character, tag = line.split(" ")

                # If character not in vocab --> Save the character
                if character not in self.vocab and self.data_type == "train":
                    self.vocab[character] = max(self.vocab.values()) + 1

                if tag not in self.tag_map and self.data_type == "train" and tag in self.tags:
                    self.tag_map[tag] = len(self.tag_map.keys())

                # Get the sentence and target tag
                sentence.append(self.vocab.get(character, 0))
                target.append(self.tag_map.get(tag, 0))

        self.input_size = len(self.vocab.values())
        return len(self.data)

    def prepare_batch(self):
        """
        prepare data for batch
        """
        index = 0
        while True:
            if index+self.batch_size >= len(self.data):
                pad_data = self.pad_data(self.data[-self.batch_size:])
                self.batch_data.append(pad_data)
                break
            else:
                pad_data = self.pad_data(self.data[index:index+self.batch_size])
                index += self.batch_size
                self.batch_data.append(pad_data)

    def pad_data(self, data: list) -> list:
        """
        If the length of the sentence is less than the max_length, then pad 0
        :param data: Input sentence list
        :return c_data: Output padded list
        """
        c_data = copy.deepcopy(data)
        for i in range(np.shape(c_data)[0]):
            if len(c_data[i][0]) >= self.max_length:
                c_data[i].append(self.max_length)
                c_data[i][0] = c_data[i][0][:self.max_length]
                c_data[i][1] = c_data[i][1][:self.max_length]
            else:
                c_data[i].append(len(c_data[i][0]))
                c_data[i][0] = c_data[i][0] + (self.max_length - len(c_data[i][0])) * [0]
                c_data[i][1] = c_data[i][1] + (self.max_length - len(c_data[i][1])) * [0]
        return c_data

    def iteration(self):
        """
        Batch iteration
        """
        idx = 0
        while True:
            yield self.batch_data[idx]
            idx += 1
            if idx > len(self.batch_data)-1:
                idx = 0

    def get_batch(self):
        """
        Get batch
        """
        for data in self.batch_data:
            yield data
