# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TextCNN(nn.Module):

    def __init__(self, config):
        super(TextCNN, self).__init__()
        self.label_num = config.num_class
        self.word_embedding_dimension = 100

        self.conv3 = nn.Conv2d(1, 1, (4, self.word_embedding_dimension))
        self.conv4 = nn.Conv2d(1, 1, (4, self.word_embedding_dimension))
        self.conv5 = nn.Conv2d(1, 1, (4, self.word_embedding_dimension))
        # self.Max3_pool = nn.MaxPool2d((self.config.sentence_max_size-100+1, 1))
        # self.Max4_pool = nn.MaxPool2d((self.config.sentence_max_size-100+1, 1))
        # self.Max5_pool = nn.MaxPool2d((self.config.sentence_max_size-100+1, 1))
        self.linear1 = nn.Sequential(nn.Linear(69, 100), nn.Linear(100, self.label_num))

    def forward(self, x, x_c):

        batch = x.shape[0]

        # Convolution
        x1 = F.relu(self.conv3(x))
        x2 = F.relu(self.conv4(x))
        x3 = F.relu(self.conv5(x))

        # Pooling
        # x1 = self.Max3_pool(x1)
        # x2 = self.Max4_pool(x2)
        # x3 = self.Max5_pool(x3)

        # capture and concatenate the features
        x = torch.cat((x1, x2, x3), -1)
        x = x.view(batch, 1, -1)
        x = torch.squeeze(x)
        # project the features to the labels
        x = self.linear1(x)

        x = x.view(-1, self.label_num)

        return x


if __name__ == '__main__':
    print('running the TextCNN...')