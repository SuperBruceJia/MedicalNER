#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import necessary Python Packages
import re
import pickle
import torch
import torch.optim as optim
from data_manager import DataManager, load_config
from model import BiLSTMCRF
from utils import *
import warnings
import numpy as np


warnings.filterwarnings("ignore")
device = torch.device("cpu")


def load_params(path: str):
    """
    Load the parameters (data)
    """
    with open(path + "data.pkl", "rb") as fopen:
        data_map = pickle.load(fopen)
    return data_map


def save_params(data, path):
    """
    The "data.pkl" is more like a mapping between Chinese Characters and Index Number
    """
    with open(path + "data.pkl", "wb") as fopen:
        pickle.dump(data, fopen)


def strQ2B(ustring):
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 12288:
            inside_code = 32
        elif inside_code >= 65281 and inside_code <= 65374:
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring


def cut_text(text, length):
    textArr = re.findall('.{' + str(length) + '}', text)
    textArr.append(text[(len(textArr) * length):])
    return textArr


class ChineseNER:
    def __init__(self, entry="train"):
        # Load some Hyper-parameters
        config = load_config()
        self.embedding_size = config.get("embedding_size")
        self.hidden_size = config.get("hidden_size")
        self.batch_size = config.get("batch_size")
        self.model_path = config.get("model_path")
        self.dropout = config.get("dropout")
        self.tags = config.get("tags")
        self.learning_rate = config.get("learning_rate")
        self.epochs = config.get("epochs")
        self.weight_decay = config.get("weight_decay")
        self.transfer_learning = config.get("transfer_learning")
        self.lr_decay_step = config.get("lr_decay_step")
        self.lr_decay_rate = config.get("lr_decay_rate")
        self.max_length = config.get("max_length")

        # Model Initialization
        self.main_model(entry)

    def main_model(self, entry):
        """
        Model Initialization
        """
        # The Training Process
        if entry == "train":
            # Training Process: read Training Data from DataManager
            self.train_manager = DataManager(batch_size=self.batch_size, data_type='train', tags=self.tags)
            self.total_size = len(self.train_manager.batch_data)

            # Read the corresponding character index (vocab) and other hyper-parameters
            data = {
                "batch_size": self.train_manager.batch_size,
                "input_size": self.train_manager.input_size,
                "vocab": self.train_manager.vocab,
                "tag_map": self.train_manager.tag_map,
            }

            save_params(data=data, path=self.model_path)

            # Build BiLSTM-CRF Model
            self.model = BiLSTMCRF(
                tag_map=self.train_manager.tag_map,
                batch_size=self.batch_size,
                vocab_size=len(self.train_manager.vocab),
                dropout=self.dropout,
                embedding_dim=self.embedding_size,
                hidden_dim=self.hidden_size,
                max_length=self.max_length
            )

            # Evaluation Process: read Dev Data from DataManager
            self.dev_size = DataManager(batch_size=1, data_type="dev", tags=self.tags).load_data()
            self.dev_manager = DataManager(batch_size=int(self.dev_size), data_type="dev", tags=self.tags)
            self.dev_batch = self.dev_manager.iteration()

            # Restore model if it exists
            self.restore_model()

        # The Testing & Inference Process
        elif entry == "predict":
            data_map = load_params(path=self.model_path)
            input_size = data_map.get("input_size")
            self.tag_map = data_map.get("tag_map")
            self.vocab = data_map.get("vocab")
            self.model = BiLSTMCRF(
                tag_map=self.tag_map,
                vocab_size=input_size,
                dropout=0.0,
                embedding_dim=self.embedding_size,
                hidden_dim=self.hidden_size,
                max_length=self.max_length
            )

            self.restore_model()

    def restore_model(self):
        """
        Restore the model if there is one
        """
        try:
            self.model.load_state_dict(torch.load(self.model_path + "params.pkl"))
            print("Model Successfully Restored!")
        except Exception as error:
            print("Model Failed to restore! {}".format(error))

    def train(self):
        """
        Training stage
        """
        model = self.model.to(device=device)

        # Transfer Learning Module
        if self.transfer_learning == True:
            keep_grad = [
                "transitions",
                "word_embeddings.weight",
                "hidden2tag.weight",
                "hidden2tag.bias",
                "linear1.weight",
                "linear1.bias",
                "linear2.weight",
                "linear2.bias"
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
        optimizer = optim.AdamW(params=filter(lambda p: p.requires_grad, model.parameters()),
                                lr=self.learning_rate,
                                weight_decay=self.weight_decay,
                                amsgrad=True)

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
                self.model.zero_grad()

                # Read sentences and tags from the batch data
                sentences, tags, length = zip(*batch)
                sentences_tensor = torch.tensor(sentences, dtype=torch.long, device=device)
                tags_tensor = torch.tensor(tags, dtype=torch.float, device=device)
                length_tensor = torch.tensor(length, dtype=torch.int64, device=device)

                # Use Negative Log-Likelihood (NLL) as Loss Function, Run the forward pass
                batch_loss = self.model.neg_log_likelihood(sentences_tensor, tags_tensor, length_tensor)
                loss = batch_loss.mean()

                progress = ("█" * int(index * 40 / self.total_size)).ljust(40)
                print("epoch [{}] |{}| {}/{}\n\t Training Loss {:.6f}".format(epoch, progress, index, self.total_size, loss))

                loss.backward()
                optimizer.step()

                # Save the model during training
                torch.save(self.model.state_dict(), self.model_path + 'params.pkl')

            self.evaluate()
            # scheduler.step()

    def evaluate(self):
        """
        Evaluation of the performance using the dev batch - dev dataset
        """
        sentences, labels, length = zip(*self.dev_batch.__next__())
        _, pre = self.model(sentences=sentences, real_length=length, lengths=None)

        sentences_tensor = torch.tensor(sentences, dtype=torch.long, device=device)
        tags_tensor = torch.tensor(pre, dtype=torch.float, device=device)
        length_tensor = torch.tensor(length, dtype=torch.int64, device=device)

        loss = self.model.neg_log_likelihood(sentences_tensor, tags_tensor, length_tensor)
        print("\t Evaluation Loss {:.6f}".format(loss.tolist()[0]))

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
                _, _, f1_prefix = entity_label_f1(tar_path=labels, pre_path=pre, length=length, tag=tag, tag_map=self.model.tag_map, prefix=prefix)
                f1_prefix_total.append(f1_prefix)

        f1_macro_tag_prefix = sum(f1_prefix_total) / len(f1_prefix_total)
        print('Tag-Label-level Macro-averaged F1 Score of the dev set is \033[1;31m%s\033[0m' % f1_macro_tag_prefix)

        ####################################################################################################################################
        # Label-level F1 score summary
        f1_prefix_total = []
        prefixes = ['B', 'I', 'E', 'S', 'O']
        for prefix in prefixes:
            _, _, f1_prefix = label_f1(tar_path=labels, pre_path=pre, length=length, tags=self.tags, tag_map=self.model.tag_map, prefix=prefix)
            f1_prefix_total.append(f1_prefix)

        f1_macro_prefix = sum(f1_prefix_total) / len(f1_prefix_total)
        print('Label-level Macro-averaged F1 Score of the dev set is \033[1;31m%s\033[0m' % f1_macro_prefix)

    def predict(self):
        """
        Prediction & Inference Stage
        :param input_str: Input Chinese sentence
        :return entities: Predicted entities
        """
        # Print model architecture
        print('\033[1;31mThe model architecture is shown below:\033[0m')
        print(self.model)
        print('\n')

        # Input one Chinese Sentence
        while True:
            input_str = input("Please input a sentence in Chinese: ")

            if len(input_str) != 0:
                # Full-width to half-width
                input_str = strQ2B(input_str)
                input_str = re.sub(pattern='。', repl='.', string=input_str)

                text = cut_text(text=input_str, length=self.max_length)

                cut_out = []
                for cuttext in text:
                    # Get the embedding vector (Input Vector) from vocab
                    input_vec = [self.vocab.get(i, 0) for i in cuttext]

                    # convert it to tensor and run the model
                    sentences = torch.tensor(input_vec).view(1, -1)

                    length = np.expand_dims(np.shape(sentences)[1], axis=0)
                    length = torch.tensor(length, dtype=torch.int64, device=device)

                    _, paths = self.model(sentences=sentences, real_length=length, lengths=None)

                    # Get the entities from the model
                    entities = []
                    for tag in self.tags:
                        tags = get_tags(paths[0], tag, self.tag_map)
                        entities += format_result(tags, cuttext, tag)

                    # Get all the entities
                    all_start = []
                    for entity in entities:
                        start = entity.get('start')
                        all_start.append([start, entity])

                    # Sort the results by the "start" index
                    sort_d = [value for index, value in sorted(enumerate(all_start), key=lambda all_start: all_start[1])]

                    if len(sort_d) == 0:
                        return print("There was no entity in this sentence!!")
                    else:
                        sort_d = np.reshape(np.array(sort_d)[:, 1], [np.shape(sort_d)[0], 1])
                        cut_out.append(sort_d)
                # return cut_out
                print(cut_out)
            else:
                return print('Invalid input! Please re-input!!\n')


# main function
if __name__ == "__main__":
    # Training the model
    cn = ChineseNER("train")
    cn.train()

    # # Predict one sentence
    # cn = ChineseNER("predict")
    # print(cn.predict())
