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
        return x


class Body_Embedding(nn.Module):

    def __init__(self, embedding_dim, hidden_size_lstm, hidden_size_att):
        super(Body_Embedding, self).__init__()

        self.gru1 = nn.GRU(embedding_dim, hidden_size_lstm, bidirectional=True, batch_first=True)  # 100,50
        self.att1 = SelfAttention(hidden_size_lstm * 2, hidden_size_att)  # 100, 100
        self.gru2 = nn.GRU(hidden_size_att, hidden_size_lstm, bidirectional=True, batch_first=True)  # 100, 50

    def forward(self, x):

        x = x.split(1, dim=1)
        x = [self.gru1(e.squeeze(1)) for e in x]
        x = [self.att1(e).unsqueeze(1) for e, _ in x]
        x = torch.cat(x, dim=1)
        x, _ = self.gru2(x)

        return x


class Comment_Embedding(nn.Module):
    def __init__(self, embedding_dim, hidden_size_lstm, hidden_size_att):
        super(Comment_Embedding, self).__init__()

        self.gru1 = nn.GRU(embedding_dim, hidden_size_lstm, bidirectional=True, batch_first=True)  # 100,50
        self.att1 = SelfAttention(hidden_size_lstm * 2, hidden_size_att)  # 100, 100

    def forward(self, x):

        x = x.split(1, dim=1)
        x = [self.gru1(e.squeeze(1)) for e in x]
        x = [self.att1(e).unsqueeze(1) for e, _ in x]
        x = torch.cat(x, dim=1)

        return x


class Co_Attention2(nn.Module):
    def __init__(self, hidden_size_att, hidden_size_co):
        super(Co_Attention2, self).__init__()

        self.Wl_weight = nn.Parameter(torch.rand(hidden_size_att, hidden_size_att))

        self.Ws_weight = nn.Parameter(torch.rand(hidden_size_co, hidden_size_att))

        self.Wc_weight = nn.Parameter(torch.rand(hidden_size_co, hidden_size_att))

        self.hs_weight = nn.Parameter(torch.rand(1, hidden_size_co))

        self.hc_weight = nn.Parameter(torch.rand(1, hidden_size_co))

    def forward(self, S, C):
        S = S.transpose(1, 2)  # batchsize*hidden_size_att*N
        C = C.transpose(1, 2)  # batchsize*hidden_size_att*T

        F_matrix = torch.tanh(C.transpose(1, 2).matmul(self.Wl_weight.matmul(S)))  # batchsize*T*N
        H_s = torch.tanh(
            self.Ws_weight.matmul(S) + self.Wc_weight.matmul(C).matmul(F_matrix))  # batchsize*hidden_size_co*N
        H_c = torch.tanh(self.Wc_weight.matmul(C) + self.Ws_weight.matmul(S).matmul(
            F_matrix.transpose(1, 2)))  # batchsize*hidden_size_co*T

        a_s = F.softmax(self.hs_weight.matmul(H_s), dim=2)  # batchsize*1*N
        a_c = F.softmax(self.hc_weight.matmul(H_c), dim=2)  # batchsize*1*T

        S = a_s.matmul(S.transpose(1, 2))  # batchsize*1*hidden_size_att
        C = a_c.matmul(C.transpose(1, 2))  # batchsize*1*hidden_size_att

        return torch.cat((S.sum(1), C.sum(1)), dim=1)  # batchsize*(hidden_size_att*2)


class dEFEND(nn.Module):
    def __init__(self, config):
        super(dEFEND, self).__init__()
        self.hidden_size_lstm = 50
        self.hidden_size_att = 100
        self.hidden_size_co = 50
        self.hidden_size_fc = 25
        self.embedding_dim = 100

        self.Body = Body_Embedding(self.embedding_dim, self.hidden_size_lstm, self.hidden_size_att)
        self.Comments = Comment_Embedding(self.embedding_dim, self.hidden_size_lstm, self.hidden_size_att)
        self.CoAtt = Co_Attention2(self.hidden_size_att, self.hidden_size_co)
        self.fc = nn.Linear(self.hidden_size_att*2, config.num_class, True)

    def forward(self, B, C):

        B = self.Body(B)
        C = self.Comments(C) # batchsize*1*hiddensiz_att
        B_and_C = self.CoAtt(B,C) # batchsize*hiddensize_att*2
        result_all = self.fc(B_and_C)

        return result_all
