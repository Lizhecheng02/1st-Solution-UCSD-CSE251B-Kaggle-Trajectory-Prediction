import torch
import torch.nn as nn


class LSTMNet(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128, output_dim=60 * 2):
        super(LSTMNet, self).__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, data):
        x = data.x.reshape(-1, 50, 50, 6)
        x = x[:, 0, :, :]

        lstm_out, _ = self.lstm(x)
        x = self.norm(lstm_out[:, -1, :])
        x = self.dropout(x)
        out = self.fc(x)
        return out.view(-1, 60, 2)


class TransformerNet(nn.Module):
    def __init__(self, input_dim=6, model_dim=128, output_dim=60 * 2, nhead=8, num_layers=4):
        super(TransformerNet, self).__init__()

        self.input_proj = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=nhead, dim_feedforward=256, dropout=0.2, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.pos_embedding = nn.Parameter(torch.randn(1, 50, model_dim))
        self.fc = nn.Linear(model_dim, output_dim)

    def forward(self, data):
        x = data.x.reshape(-1, 50, 50, 6)
        x = x[:, 0, :, :]

        x = self.input_proj(x)
        x = x + self.pos_embedding

        x = self.transformer_encoder(x)
        x = x[:, -1, :]

        out = self.fc(x)
        return out.view(-1, 60, 2)
