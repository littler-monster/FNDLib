import torch
from torch import nn
import torch.nn.functional as F


class SelfAttention(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(SelfAttention, self).__init__()
        self.W = nn.Linear(input_size, hidden_size, True)
        self.u = nn.Linear(hidden_size, 1)

    def forward(self, x):
        u = torch.tanh(self.W(x))

        a = F.softmax(self.u(u), dim=1)

        x = a.mul(x)
        x = x.sum(1)
        return x, a


class HAN(nn.Module):

    def __init__(self, config):
        super(HAN, self).__init__()

        self.embedding_dim = 100
        self.hidden_size_gru = 25
        self.hidden_size_att = 50
        self.hidden_size_fc = 25

        self.gru1 = nn.GRU(self.embedding_dim, self.hidden_size_gru, bidirectional=True, batch_first=True)  # 100,25
        self.att1 = SelfAttention(self.hidden_size_gru * 2, self.hidden_size_att)  # 50, 50

        self.gru2 = nn.GRU(self.hidden_size_att, self.hidden_size_gru, bidirectional=True,
                           batch_first=True)  # 50, 25
        self.att2 = SelfAttention(self.hidden_size_gru * 2, self.hidden_size_att)  # 50, 50

        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size_att, config.num_class, True),  # 50, 25
        )

    def forward(self, x, x_c):

        x = x.float()

        x = x.split(1, dim=1)

        x = [self.gru1(e.squeeze(1)) for e in x]
        x = [self.att1(e)[0].unsqueeze(1) for e, _ in x]
        x = torch.cat(x, dim=1)
        x, _ = self.gru2(x)
        x, weight = self.att2(x)
        x = self.fc(x)

        return x
