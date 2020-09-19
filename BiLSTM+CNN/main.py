#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import necessary Python Packages
import jieba
import pickle
import codecs
import torch
import torch.optim as optim
from model import BiLSTMCRF
from data_manager import DataManager, load_config
from utils import *
import warnings


warnings.filterwarnings("ignore")
device = torch.device("cpu")


def save_params(data, path):
    """
    Save all the parameters: batch_size, input_size, char_vocab, tag_map
    """
    with open(path + "data.pkl", "wb") as fopen:
        pickle.dump(data, fopen)


def load_params(path):
    """
    Load the parameters during prediction
    """
    with open(path + "data.pkl", "rb") as fopen:
        char_map = pickle.load(fopen)
    return char_map


def Q2B(uchar):
    inside_code = ord(uchar)
    if inside_code == 0x3000:
        inside_code = 0x0020
    else:
        inside_code -= 0xfee0
    if inside_code < 0x0020 or inside_code > 0x7e:
        return uchar
    return chr(inside_code)


def stringQ2B(ustring):
    return "".join([Q2B(uchar) for uchar in ustring])


class ChineseNER:
    def __init__(self, entry="train"):
        # Load Hyper-parameters
        config = load_config()
        self.model_path = config.get("model_path")
        self.epochs = config.get("epochs")
        self.batch_size = config.get("batch_size")
        self.learning_rate = config.get("learning_rate")
        self.weight_decay = config.get("weight_decay")
        self.dropout = config.get("dropout")
        self.hidden_size = config.get("hidden_size")
        self.char_num = config.get("char_num")
        self.char_dim = config.get("char_dim")
        self.word_dim = config.get("word_dim")
        self.word_num = config.get("word_num")
        self.tags = config.get("tags")
        self.transfer_learning = config.get("transfer_learning")
        self.lr_decay_step = config.get("lr_decay_step")
        self.lr_decay_rate = config.get("lr_decay_rate")

        # Load main model
        self.main_model(entry)

    def main_model(self, entry):
        # The Training Process
        if entry == "train":
            # Training Process: read Training Data from DataManager
            self.train_manager = DataManager(batch_size=self.batch_size, data_type='train', tags=self.tags)
            self.total_size = len(self.train_manager.batch_data)

            # Read the corresponding character index (vocab) and other hyper-parameters
            saved_data = {
                "batch_size": self.train_manager.batch_size,
                "input_size": self.train_manager.input_size,
                "char_vocab": self.train_manager.char_vocab,
                "tag_map": self.train_manager.tag_map,
            }
            save_params(data=saved_data, path=self.model_path)

            # Evaluation Process: read Dev Data from DataManager
            self.dev_size = DataManager(batch_size=1, data_type="dev", tags=self.tags).load_char_data()
            self.dev_manager = DataManager(batch_size=int(self.dev_size), data_type="dev")
            self.dev_batch = self.dev_manager.iteration()

            # Build BiLSTM-CRF Model
            self.model = BiLSTMCRF(
                tag_map=self.train_manager.tag_map,
                batch_size=self.batch_size,
                vocab_size=len(self.train_manager.char_vocab),
                dropout=self.dropout,
                word_num=self.word_num,
                word_dim=self.word_dim,
                char_num=self.char_num,
                char_dim=self.char_dim,
                hidden_dim=self.hidden_size,
            )

            # Restore model if it exists
            self.restore_model()

        # The Inference Process
        elif entry == "predict":
            data = load_params(path=self.model_path)
            input_size = data.get("input_size")
            self.tag_map = data.get("tag_map")
            self.vocab = data.get("char_vocab")
            self.model = BiLSTMCRF(
                tag_map=self.tag_map,
                vocab_size=input_size,
                dropout=1.0,
                word_num=self.word_num,
                word_dim=self.word_dim,
                char_num=self.char_num,
                char_dim=self.char_dim,
                hidden_dim=self.hidden_size,
            )
            self.restore_model()

    def restore_model(self):
        """
        Restore and load the model
        """
        try:
            self.model.load_state_dict(torch.load(self.model_path + "params.pkl"))
            print("Model Successfully Restored!!")
        except Exception as error:
            print("Model Failed to restore!!")

    def train(self):
        model = self.model.to(device=device)

        # Transfer Learning Module
        if self.transfer_learning == True:
            keep_grad = [
                "transitions",
                "char_embed.weight",
                "linear_lstm.weight",
                "linear_lstm.bias",
                "linear_cnn.weight",
                "linear_cnn.bias",
                "hidden2tag.weight",
                "hidden2tag.bias"
            ]

            for name, value in model.named_parameters():
                if name in keep_grad:
                    value.requires_grad = True
                else:
                    value.requires_grad = False
        else:
            for name, value in model.named_parameters():
                value.requires_grad = True

        # Use Adam Optimizer
        optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=self.learning_rate, weight_decay=self.weight_decay)

        # Learning Rate Decay
        # scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=self.lr_decay_step, gamma=self.lr_decay_rate)

        # Print model architecture
        print('\033[1;31mThe model architecture is shown below:\033[0m')
        print(model)
        print('\n')

        # Print model parameters
        print('\033[1;31mThe model\'s parameters are shown below:\033[0m')
        for name, value in model.named_parameters():
            print("Name: \033[1;31m{0}\033[0m, "
                  "Parameter Size: \033[1;36m{1}\033[0m, "
                  "Gradient: \033[1;35m{2}\033[0m".format(name, value.size(), value.requires_grad))
        print('\n')

        for epoch in range(1, self.epochs+1):
            index = 0
            for batch in self.train_manager.get_batch():
                index += 1
                # Clear gradients before training
                model.zero_grad()

                # Read sentences and tags from the batch data
                chars, tags, words, len_char = zip(*batch)
                chars_tensor = torch.tensor(chars, dtype=torch.long, device=device)
                tags_tensor = torch.tensor(tags, dtype=torch.float, device=device)
                words_tensor = torch.tensor(words, dtype=torch.float, device=device)
                leng_char = torch.tensor(len_char, dtype=torch.int64, device=device)

                loss = model.neg_log_likelihood(characters=chars_tensor, tags=tags_tensor, length=leng_char, words=words_tensor)
                progress = ("█" * int(index * 40 / self.total_size)).ljust(40)
                print("epoch [{}] |{}| {}/{}\t Batch Loss {:.6f}".format(epoch, progress, index, self.total_size, loss.tolist()[0]))

                loss.backward()
                optimizer.step()
                torch.save(model.state_dict(), self.model_path + 'params.pkl')

            self.evaluate()
            # scheduler.step()

    def evaluate(self):
        """
        Evaluation of the performance using the development set
        """
        model = self.model.to(device)

        chars, labels, words, len_chars = zip(*self.dev_batch.__next__())
        chars_tensor = torch.tensor(chars, dtype=torch.long, device=device)
        words_tensor = torch.tensor(words, dtype=torch.float, device=device)
        len_char_tensor = torch.tensor(len_chars, dtype=torch.int64, device=device)

        # Run the Forward pass of the model
        _, pre = model(characters=chars_tensor, words=words_tensor, len_char=len_chars)
        pre_tensor = torch.tensor(pre, dtype=torch.int, device=device)

        ####################################################################################################################################
        # Loss on the dev set
        loss = model.neg_log_likelihood(characters=chars_tensor, tags=pre_tensor, length=len_char_tensor, words=words_tensor)
        print("\t Evaluation Loss on the dev set {:.6f}".format(loss.tolist()[0]))

        ####################################################################################################################################
        print('Start to evaluate on the dev set: ')
        # Tag-level F1 score summary (w.r.t. each tag)
        tag_f1_total = []
        for tag in self.tags:
            _, _, f1_tag = tag_f1(tar_path=labels, pre_path=pre, tag=tag, tag_map=self.model.tag_map)
            tag_f1_total.append(f1_tag)
        tag_macro_f1 = sum(tag_f1_total) / len(tag_f1_total)
        print('Tag-level Macro-averaged F1 Score of the dev set is \033[1;31m%s\033[0m' % tag_macro_f1)

        # Tag-level Micro-averaged F1 Score
        _, _, f1_Micro_tag = tag_micro_f1(tar_path=labels, pre_path=pre, tags=self.tags, tag_map=self.model.tag_map)
        print('Tag-level Micro-averaged F1 Score of the dev set is \033[1;35m%s\033[0m' % f1_Micro_tag)

        ####################################################################################################################################
        # Tag-level with Label-level F1 score summary
        f1_prefix_total = []
        prefixes = ['B', 'I', 'E', 'S']
        for tag in self.tags:
            for prefix in prefixes:
                _, _, f1_prefix = entity_label_f1(tar_path=labels,
                                                  pre_path=pre,
                                                  length=len_chars,
                                                  tag=tag,
                                                  tag_map=self.model.tag_map,
                                                  prefix=prefix)
                f1_prefix_total.append(f1_prefix)

        f1_macro_tag_prefix = sum(f1_prefix_total) / len(f1_prefix_total)
        print('Tag-Label-level Macro-averaged F1 Score of the dev set is \033[1;31m%s\033[0m' % f1_macro_tag_prefix)

        ####################################################################################################################################
        # Label-level F1 score summary
        f1_prefix_total = []
        prefixes = ['B', 'I', 'E', 'S', 'O']
        for prefix in prefixes:
            _, _, f1_prefix = label_f1(tar_path=labels,
                                       pre_path=pre,
                                       length=len_chars,
                                       tags=self.tags,
                                       tag_map=self.model.tag_map,
                                       prefix=prefix)
            f1_prefix_total.append(f1_prefix)

        f1_macro_prefix = sum(f1_prefix_total) / len(f1_prefix_total)
        print('Label-level Macro-averaged F1 Score of the dev set is \033[1;31m%s\033[0m' % f1_macro_prefix)

    def load_word_vector(self):
        """
        Load pre-trained word vectors
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
    
    def pad_char_data(self, data: list) -> list:
        """
        Pad character data
        """
        c_data = copy.deepcopy(data)
        if np.shape(c_data)[0] < self.char_num:
            c_data = c_data + (self.char_num - np.shape(c_data)[0]) * [0]
        else:
            c_data = c_data[:self.char_num]
        c_data = np.expand_dims(c_data, axis=0)
        return c_data

    def pad_word_data(self, data: list) -> list:
        """
        Pad word data
        """
        c_data = copy.deepcopy(data)
        if len(c_data) <= self.word_num:
            c_data = c_data + (self.word_num - len(c_data)) * [[0] * self.word_dim]
        else:
            c_data = c_data[:self.word_num, :]
        c_data = np.reshape(c_data, [np.shape(c_data)[0] * np.shape(c_data)[1]])
        c_data = np.expand_dims(c_data, axis=0)
        return c_data

    def predict(self):
        """
        Prediction & Inference Stage
        """
        self.pre_trained = self.load_word_vector()

        while True:
            input_str = input("Please input a sentence (in Chinese): ")

            # Get character embedding
            char_vec = [self.vocab.get(i, 0) for i in input_str]
            char_tensor = np.reshape(char_vec, [-1]).tolist()
            len_char = len(char_tensor)
            char_tensor = np.array(self.pad_char_data(char_tensor)).tolist()
            char_tensor = torch.tensor(char_tensor, dtype=torch.long, device=device)

            # Get word embedding
            embed_words = []
            words = jieba.lcut(input_str, HMM=True)
            for i in words:
                vec = self.pre_trained.get(i)
                if str(type(vec)) != "<class 'NoneType'>":
                    embed_words.append(vec)
            word_tensor = np.array(self.pad_word_data(embed_words)).tolist()
            word_tensor = torch.tensor(word_tensor, dtype=torch.float, device=device)

            # Run the model
            _, paths = self.model(characters=char_tensor, words=word_tensor, len_char=len_char)

            # Get the entities and format the results
            entities = []
            for tag in self.tags:
                tags = get_tags(path=paths[0], tag=tag, tag_map=self.tag_map)
                entities += format_result(result=tags, text=input_str, tag=tag)
            print(entities)


if __name__ == "__main__":
    # Training the model
    cn = ChineseNER("train")
    cn.train()

    # # # Predict one sentence
    # cn = ChineseNER("predict")
    # print(cn.predict())
