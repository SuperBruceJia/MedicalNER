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
            batch_size=16,
            vocab_size=20,
            hidden_dim=128,
            dropout=0.0,
            word_num=100,
            word_dim=128,
            char_num=200,
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
        # Matrix of transition parameters. Entry i,j is the score of transitioning *to* i *from* j
        self.transitions = nn.Parameter(torch.randn(self.tag_size, self.tag_size))
        self.transitions.data[self.tag_map[self.start_tag], :] = -10000.
        self.transitions.data[:, self.tag_map[self.stop_tag]] = -10000.

        self.tanh = nn.Tanh()
        self.LeakyReLU = nn.LeakyReLU()
        self.softplus = nn.Softplus()
        self.gelu = nn.GELU()
        self.Dropout = nn.Dropout(p=self.dropout)

        ####################################################################################################################################
        # Left side Bi-LSTM model --> Character-level Embedding
        self.char_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=self.char_dim)

        # The model of Bi-LSTM
        self.char_lstm = nn.LSTM(input_size=self.char_dim,
                                 hidden_size=self.hidden_dim // 2,
                                 num_layers=1,
                                 bidirectional=True,
                                 batch_first=True,
                                 dropout=self.dropout,
                                 bias=True)

        self.char_linear_lstm = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim, bias=True)
        
        ####################################################################################################################################
        # Right side CNN model --> Word-level Embedding
        # The model of Bi-LSTM
        self.word_lstm = nn.LSTM(input_size=self.word_dim,
                                 hidden_size=self.hidden_dim // 2,
                                 num_layers=1,
                                 bidirectional=True,
                                 batch_first=True,
                                 dropout=self.dropout,
                                 bias=True)

        self.word_linear_lstm = nn.Linear(in_features=self.hidden_dim, out_features=self.char_num, bias=True)

        ####################################################################################################################################
        self.linear_1 = nn.Linear(in_features=712, out_features=512, bias=True)
        self.linear_2 = nn.Linear(in_features=512, out_features=256, bias=True)

        ####################################################################################################################################
        # Hidden to tag
        self.hidden2tag = nn.Linear(in_features=256, out_features=self.tag_size, bias=True)

    # len_char: Real length of characters (sentences)
    # len_word: Real length of words
    def prediction(self, characters, len_char, words, len_word):
        ####################################################################################################################################
        # Left side Bi-LSTM model --> Character-level Embedding
        char_vec = self.char_embedding(characters)
        char_vec = self.Dropout(char_vec)
        char_vec = char_vec.view(self.batch_size, self.char_num, self.char_dim)

        packed_char = nn.utils.rnn.pack_padded_sequence(char_vec, len_char, batch_first=True, enforce_sorted=False)
        char_out, (_, _) = self.char_lstm(packed_char)
        unpacked_char, _ = nn.utils.rnn.pad_packed_sequence(char_out, batch_first=True, padding_value=0.0, total_length=self.char_num)
        unpacked_char = unpacked_char.view(self.batch_size, -1, self.hidden_dim)

        char_map = self.softplus(self.char_linear_lstm(unpacked_char))
        char_map = self.Dropout(char_map)

        ####################################################################################################################################
        # Right side CNN model --> Word-level Embedding
        words_reshaped = words.view(self.batch_size, self.word_num, self.word_dim)

        packed_word = nn.utils.rnn.pack_padded_sequence(words_reshaped, len_word, batch_first=True, enforce_sorted=False)
        word_out, (_, _) = self.word_lstm(packed_word)
        unpacked_word, _ = nn.utils.rnn.pad_packed_sequence(word_out, batch_first=True, padding_value=0.0, total_length=self.word_num)
        unpacked_word = unpacked_word.view(self.batch_size, -1, self.hidden_dim)

        word_map = self.softplus(self.word_linear_lstm(unpacked_word))
        word_map = self.Dropout(word_map)
        word_map = word_map.view(self.batch_size, self.char_num, -1)

        ####################################################################################################################################
        # Concatenate the outputs of LSTM and CNN
        cat_lstm_cnn = torch.cat([char_map, word_map], dim=2)

        # Linear Layer
        linear_out = self.softplus(self.linear_1(cat_lstm_cnn))
        linear_out = self.Dropout(linear_out)

        linear_out = self.softplus(self.linear_2(linear_out))
        linear_out = self.Dropout(linear_out)

        # Hidden space to tag space
        logits = self.softplus(self.hidden2tag(linear_out))
        return logits

    def neg_log_likelihood(self, characters, tags, len_char, words, len_word):
        """
        Negative Log-Likelihood (NLL) Loss Function
        """
        self.batch_size = characters.size(0)
        logits = self.prediction(characters=characters, len_char=len_char, words=words, len_word=len_word)
        real_path_score = torch.zeros(1, device=device)
        total_score = torch.zeros(1, device=device)

        for logit, tag, leng in zip(logits, tags, len_char):
            logit = logit[:leng.to(torch.int)]
            tag = tag[:leng.to(torch.int)]
            real_path_score += self.real_path_score(logit, tag)
            total_score += self.total_score(logit, tag)
        return total_score - real_path_score

    def forward(self, characters, words, len_char=None, len_word=None):
        characters = torch.tensor(characters, dtype=torch.long, device=device)
        words = torch.tensor(words, dtype=torch.float, device=device)

        lengths = [i.size(-1) for i in characters]
        self.batch_size = characters.size(0)
        logits = self.prediction(characters=characters, len_char=len_char, words=words, len_word=len_word)

        scores = []
        paths = []
        for logit, leng in zip(logits, lengths):
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
            emission_score = logit[label[index + 1]]
            transition_score = self.transitions[label[index], label[index + 1]]
            score += emission_score + transition_score

        # Add the final Stop Tag, the final transition score
        score += self.transitions[label[-1], self.tag_map[self.stop_tag]]
        return score

    def total_score(self, logits, label):
        """
        Calculate the total CRF Score
        """
        previous = torch.full((1, self.tag_size), 0, device=device)
        for index in range(len(logits)):
            previous = previous.expand(self.tag_size, self.tag_size).t()
            obs = logits[index].view(1, -1).expand(self.tag_size, self.tag_size)
            scores = previous + obs + self.transitions
            previous = log_sum_exp(scores)
        previous = previous + self.transitions[:, self.tag_map[self.stop_tag]]
        total_scores = log_sum_exp(previous.t())[0]
        return total_scores

    def viterbi_decode(self, logits):
        backpointers = []
        trellis = torch.zeros(logits.size(), device=device)
        backpointers = torch.zeros(logits.size(), dtype=torch.long, device=device)
        trellis[0] = logits[0]
        for t in range(1, len(logits)):
            v = trellis[t - 1].unsqueeze(1).expand_as(self.transitions) + self.transitions
            trellis[t] = logits[t] + torch.max(v, 0)[0]
            backpointers[t] = torch.max(v, 0)[1]
        viterbi = [torch.max(trellis[-1], -1)[1].cpu().tolist()]
        backpointers = backpointers.numpy()
        for bp in reversed(backpointers[1:]):
            viterbi.append(bp[viterbi[-1]])
        viterbi.reverse()

        viterbi_score = torch.max(trellis[-1], 0)[0].tolist()
        return viterbi_score, viterbi
