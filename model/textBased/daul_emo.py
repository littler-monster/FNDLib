import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.W = nn.Parameter(torch.randn(input_dim))
        nn.init.normal_(self.W)  
    
    def forward(self, x):
        # x 的形状: (batch_size, seq_length, input_dim)
        # (batch_size, seq_length, input_dim) * (input_dim,) -> (batch_size, seq_length)
        e = torch.exp(torch.tanh(torch.sum(x * self.W, dim=-1)))
        # 归一化注意力权重
        a = e / torch.sum(e, dim=1, keepdim=True) # (batch_size, seq_length)
        weight_input = x  * a.unsqueeze(-1)
        return torch.sum(weight_input, dim=1)

class HSA_BLSTM(nn.Module):
    def __init__(self, con_embedding_matrix, com_embedding_matrix, emotion_dim=0,
                 hidden_units=32):
        super(HSA_BLSTM, self).__init__()

        # Content embedding
        vocab_size_con, embedding_dim_con = con_embedding_matrix.shape
        self.embedding_con = nn.Embedding.from_pretrained(torch.tensor(con_embedding_matrix, dtype=torch.float32), freeze=True)

        # Comments embedding
        vocab_size_com, embedding_dim_com = com_embedding_matrix.shape
        self.embedding_com = nn.Embedding.from_pretrained(torch.tensor(com_embedding_matrix, dtype=torch.float32), freeze=True)
        
        # Hierarchical LSTM for content
        self.hidden_units = hidden_units
        self.lstm_content = nn.LSTM(embedding_dim_con, hidden_units, bidirectional=True, batch_first=True)
        self.att_content = SelfAttention(hidden_units * 2)

        # Hierarchical LSTM for comments
        self.lstm_comment_word = nn.LSTM(embedding_dim_com, hidden_units, bidirectional=True, batch_first=True)
        self.att_comment_word = SelfAttention(hidden_units * 2)

        self.lstm_comment_post = nn.LSTM(hidden_units * 2, hidden_units, bidirectional=True, batch_first=True)
        self.att_comment_post = SelfAttention(hidden_units * 2)

        self.lstm_comment_sub = nn.LSTM(hidden_units * 2, hidden_units, bidirectional=True, batch_first=True)
        self.att_comment_sub = SelfAttention(hidden_units * 2)

        #Dense layers
        self.fc1 = nn.Linear(hidden_units * 4 + emotion_dim, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, con_input, com_input, emotion_input=None):
        #Content processing
        con_emb = self.embedding_con(con_input)
        con_lstm_out, _ = self.lstm_content(con_emb)
        con_vec = self.att_content(con_lstm_out)

        #Comments processing (hierarchical)
        batch_size, sub_event_num, post_num, com_steps = com_input.size()
        com_input = com_input.view(-1, com_steps)
        com_emb = self.embedding_com(com_input)
        com_emb = com_emb.view(batch_size, sub_event_num, post_num, com_steps, -1)

        com_lstm_out_word = self.lstm_comment_word(com_emb.view(-1, com_steps, -1))[0]
        com_att_out_word = self.att_comment_word(com_lstm_out_word).view(batch_size, sub_event_num, post_num, -1)

        com_lstm_out_post = self.lstm_comment_post(com_att_out_word).view(-1, post_num, -1)[0]
        com_att_out_post = self.att_comment_post(com_lstm_out_post).view(batch_size, sub_event_num, -1)[0]

        com_lstm_out_sub = self.lstm_comment_sub(com_att_out_post).view(-1, sub_event_num, -1)[0]
        com_vec = self.att_comment_sub(com_lstm_out_sub)

        # Concatenate content and comment vectors
        semantics = torch.cat((con_vec, com_vec), dim=1)

        if emotion_input is not None:
            semantics - torch.cat((semantics, emotion_input), dim=1)

        x = F.relu(self.fc1(semantics))
        output = F.softmax(self.fc2(x), dim=1)

        return output

