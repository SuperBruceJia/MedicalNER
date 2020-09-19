#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import necessary Python Packages
import torch
from torch import nn


device = torch.device("cpu")


def log_sum_exp(vec):
    """
    log(sum(exp(x))) Function
    """
    max_score = torch.max(vec, 0)[0].unsqueeze(0)
    max_score_broadcast = max_score.expand(vec.size(1), vec.size(1))
    result = max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast), 0)).unsqueeze(0)
    return result.squeeze(1)


class BiLSTMCRF(nn.Module):
    def __init__(
        self,
        tag_map={"O": 0, "START": 4, "STOP": 5},
        batch_size=256,
        vocab_size=20,
        hidden_dim=128,
        dropout=1.0,
        word_num=30,
        word_dim=128,
        char_num=300,
        char_dim=30,
        start_tag="START",
        stop_tag="STOP"
    ):
        super(BiLSTMCRF, self).__init__()
        self.word_num = word_num
        self.word_dim = word_dim
        self.char_num = char_num
        self.char_dim = char_dim
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.tag_size = len(tag_map)
        self.tag_map = tag_map
        self.start_tag = start_tag
        self.stop_tag = stop_tag

        ####################################################################################################################################
        # Matrix of transition parameters.
        self.transitions = nn.Parameter(torch.randn(self.tag_size, self.tag_size, device=device))
        self.transitions.data[self.tag_map[self.start_tag], :] = -10000.
        self.transitions.data[:, self.tag_map[self.stop_tag]] = -10000.

        ####################################################################################################################################
        self.Leaky_ReLu = nn.LeakyReLU()
        self.softplus = nn.Softplus()
        self.gelu = nn.GELU()
        self.Dropout = nn.Dropout(p=self.dropout)
        self.Dropout_2D = nn.Dropout2d(p=self.dropout)

        ####################################################################################################################################
        # Left side Bi-LSTM model --> Character-level Embedding
        self.char_embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=self.char_dim)

        # The model of Bi-LSTM
        self.lstm = nn.LSTM(input_size=self.char_dim,
                            hidden_size=self.hidden_dim // 2,
                            num_layers=1,
                            bidirectional=True,
                            batch_first=True,
                            dropout=self.dropout)

        self.linear_lstm = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim, bias=True)

        ####################################################################################################################################
        # Right side CNN model --> Word-level Embedding
        self.fm1, self.fm2, self.fm3 = 16, 32, 64
        self.temp_size = 12

        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=self.fm1, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.bn_1 = nn.BatchNorm2d(self.fm1)
        self.pool_1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)

        self.conv_2 = nn.Conv2d(in_channels=self.fm1, out_channels=self.fm2, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.bn_2 = nn.BatchNorm2d(self.fm2)
        self.pool_2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)

        self.conv_3 = nn.Conv2d(in_channels=self.fm2, out_channels=self.fm3, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.bn_3 = nn.BatchNorm2d(self.fm3)
        self.pool_3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)

        self.linear_cnn = nn.Linear(in_features=self.temp_size, out_features=self.char_num, bias=True)

        ####################################################################################################################################
        # self.linear = nn.Linear(in_features=2 * self.hidden_dim, out_features=100, bias=True)

        self.linear_1 = nn.Linear(in_features=2 * self.hidden_dim, out_features=100, bias=True)
        self.linear_2 = nn.Linear(in_features=100, out_features=94, bias=True)

        ####################################################################################################################################
        # This Linear Layer maps the output of the LSTM + CNN into tag space (tag_size).
        self.hidden2tag = nn.Linear(in_features=94, out_features=self.tag_size, bias=True)

    def prediction(self, characters, length, words):
        ####################################################################################################################################
        # Left side Bi-LSTM model --> Character-level Embedding
        self.char_length = characters.shape[1]
        chars_vec = self.char_embed(characters)
        chars_vec = chars_vec.view(self.batch_size, self.char_length, self.char_dim)

        packed = nn.utils.rnn.pack_padded_sequence(chars_vec, length, batch_first=True, enforce_sorted=False)
        lstm_out, (_, _) = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True, total_length=self.char_length)

        lstm_mapping = self.Leaky_ReLu(self.linear_lstm(lstm_out))
        lstm_mapping = self.Dropout(lstm_mapping)

        ####################################################################################################################################
        # Right side CNN model --> Word-level Embedding
        words_reshape = words.view(self.batch_size, 1, self.word_num, self.word_dim)

        cnn_out = self.conv_1(words_reshape)
        cnn_out = self.bn_1(cnn_out)
        cnn_out = self.Leaky_ReLu(cnn_out)
        cnn_out = self.Dropout_2D(cnn_out)
        cnn_out = self.pool_1(cnn_out)

        cnn_out = self.conv_2(cnn_out)
        cnn_out = self.bn_2(cnn_out)
        cnn_out = self.Leaky_ReLu(cnn_out)
        cnn_out = self.Dropout_2D(cnn_out)
        cnn_out = self.pool_2(cnn_out)

        cnn_out = self.conv_3(cnn_out)
        cnn_out = self.bn_3(cnn_out)
        cnn_out = self.Leaky_ReLu(cnn_out)
        cnn_out = self.Dropout_2D(cnn_out)
        cnn_out = self.pool_3(cnn_out)

        cnn_flatten = cnn_out.view(self.batch_size, self.hidden_dim, -1)
        cnn_mapping = self.Leaky_ReLu(self.linear_cnn(cnn_flatten))
        cnn_mapping = self.Dropout(cnn_mapping)
        cnn_mapping = cnn_mapping.view(self.batch_size, self.char_num, -1)

        ####################################################################################################################################
        # Concatenate the outputs of LSTM and CNN
        cat_lstm_cnn = torch.cat([lstm_mapping, cnn_mapping], dim=2)

        # Linear Layer
        linear_out = self.Leaky_ReLu(self.linear_1(cat_lstm_cnn))
        linear_out = self.Dropout(linear_out)

        linear_out = self.Leaky_ReLu(self.linear_2(linear_out))
        linear_out = self.Dropout(linear_out)

        # Hidden space to tag space
        logits = self.Leaky_ReLu(self.hidden2tag(linear_out))
        return logits

    def neg_log_likelihood(self, characters, tags, length, words):
        """
        Negative Log-Likelihood (NLL) Loss Function
        """
        self.batch_size = characters.size(0)

        # Get the output tag_size tensor from the Linear Layer
        logits = self.prediction(characters=characters.to(device), length=length.to(device), words=words.to(device))
        real_path_score = torch.zeros(1, device=device)
        total_score = torch.zeros(1, device=device)

        for logit, tag, leng in zip(logits, tags, length):
            logit = logit[:leng.to(torch.int)].to(device)
            tag = tag[:leng.to(torch.int)].to(device)
            real_path_score += self.real_path_score(logit, tag).to(device)
            total_score += self.total_score(logit, tag).to(device)
        return total_score - real_path_score

    def forward(self, characters, words, len_char=None):
        characters = torch.tensor(characters, dtype=torch.long, device=device)
        words = torch.tensor(words, dtype=torch.float, device=device)
        pad_lengths = [i.size(-1) for i in characters]
        self.batch_size = characters.size(0)
        logits = self.prediction(characters=characters, length=len_char, words=words)

        scores = []
        paths = []
        for logit, leng in zip(logits, pad_lengths):
            logit = logit[:leng]
            score, path = self.viterbi_decode(logit)
            scores.append(score)
            paths.append(path)
        return scores, paths

    def real_path_score(self, logits, label):
        """
        Calculate Real Path Score
        """
        score = torch.zeros(1, device=device)
        label = torch.cat([torch.tensor([self.tag_map[self.start_tag]], dtype=torch.long, device=device), label.to(torch.long)])
        for index, logit in enumerate(logits):
            emission_score = logit[label[index + 1]].to(device)
            transition_score = self.transitions[label[index], label[index + 1]].to(device)
            score += emission_score + transition_score

        # Add the final Stop Tag, the final transition score
        score += self.transitions[label[-1].to(device), self.tag_map[self.stop_tag]]
        return score

    def total_score(self, logits, label):
        """
        Calculate the total Score
        """
        obs = []
        previous = torch.full((1, self.tag_size), 0)
        for index in range(len(logits)):
            previous = previous.expand(self.tag_size, self.tag_size).t()
            obs = logits[index].view(1, -1).expand(self.tag_size, self.tag_size)
            scores = previous + obs.to(device) + self.transitions
            previous = log_sum_exp(scores)

        previous = previous + self.transitions[:, self.tag_map[self.stop_tag]]
        total_scores = log_sum_exp(previous.t())[0]
        return total_scores

    def viterbi_decode(self, logits):
        """
        Viterbi Algorithm - Find the optimal value of the CRF model
        """
        backpointers = []
        trellis = torch.zeros(logits.size(), device=device)
        backpointers = torch.zeros(logits.size(), dtype=torch.long, device=device)
        
        trellis[0] = logits[0]
        for t in range(1, len(logits)):
            v = trellis[t - 1].unsqueeze(1).expand_as(self.transitions) + self.transitions
            trellis[t] = logits[t].to(device) + torch.max(v, 0)[0].to(device)
            backpointers[t] = torch.max(v, 0)[1]
        viterbi = [torch.max(trellis[-1], -1)[1].tolist()]
        for bp in reversed(backpointers[1:]):
            viterbi.append(bp[viterbi[-1]])
        viterbi.reverse()
        viterbi_score = torch.max(trellis[-1], 0)[0].tolist()
        return viterbi_score, viterbi
