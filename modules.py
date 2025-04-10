import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return x


class AgentTypeEmbedding(nn.Module):
    def __init__(self, num_types, d_model):
        super(AgentTypeEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_types, d_model)

    def forward(self, agent_types):
        return self.embedding(agent_types)


class SocialTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(SocialTransformerEncoderLayer, self).__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )


class SocialLSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=4, dropout=0.1):
        super(SocialLSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

    def forward(self, x):
        outputs, (h_n, c_n) = self.lstm(x)
        return outputs, (h_n, c_n)


class LSTMTrajectoryDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=4, dropout=0.1):
        super(LSTMTrajectoryDecoder, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        self.output_fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden=None):
        outputs, hidden = self.lstm(x, hidden)
        predictions = self.output_fc(outputs)

        return predictions, hidden


class TransformerTrajectoryDecoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout, max_len):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.output_fc = nn.Linear(d_model, 2)

    def forward(self, tgt, memory, tgt_mask=None):
        tgt = self.positional_encoding(tgt)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask)
        return self.output_fc(output)
