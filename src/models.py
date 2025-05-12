import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv


class LSTMNet(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128, num_layers=2, output_dim=60 * 2):
        super(LSTMNet, self).__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
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
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=nhead, dim_feedforward=model_dim * 4, dropout=0.2, batch_first=True)
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


class InteractionTransformer(nn.Module):
    def __init__(
        self,
        input_dim=6,
        model_dim=64,
        hist_len=50,
        num_agents=50,
        pred_len=60,
        nhead=8,
        time_layers=4,
        agent_layers=4,
        dropout=0.2
    ):
        super().__init__()
        self.hist_len = hist_len
        self.num_agents = num_agents
        self.pred_len = pred_len

        self.input_proj = nn.Linear(input_dim, model_dim)
        self.time_pos_embedding = nn.Parameter(torch.randn(1, hist_len, model_dim))
        time_enc_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.time_encoder = nn.TransformerEncoder(time_enc_layer, num_layers=time_layers)

        self.agent_pos_embedding = nn.Parameter(torch.randn(1, num_agents, model_dim))
        agent_enc_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.agent_encoder = nn.TransformerEncoder(agent_enc_layer, num_layers=agent_layers)

        self.fc = nn.Linear(model_dim, pred_len * 2)

    def forward(self, data: Data):
        x = data.x
        mask = data.mask
        if x.dim() == 3:
            total_agents, T, D = x.shape
            if total_agents == self.num_agents:
                x = x.unsqueeze(0)
                mask = mask.unsqueeze(0)
            else:
                B = mask.shape[0] // self.num_agents
                x = x.view(B, self.num_agents, T, D)
                mask = mask.view(B, self.num_agents)
        B, N, T, D = x.shape

        x_flat = x.view(B * N, T, D)
        h = self.input_proj(x_flat) + self.time_pos_embedding
        h = self.time_encoder(h)
        h_last = h[:, -1, :]
        agent_embed = h_last.view(B, N, -1)

        pad_mask = ~mask
        a = agent_embed + self.agent_pos_embedding
        a = self.agent_encoder(a, src_key_padding_mask=pad_mask)

        ego_emb = a[:, 0, :]
        out = self.fc(ego_emb)
        return out.view(B, self.pred_len, 2)


class InteractionTransformerWithGAT(nn.Module):
    def __init__(
        self,
        input_dim=6,
        model_dim=128,
        hist_len=50,
        num_agents=50,
        pred_len=60,
        nhead=8,
        time_layers=4,
        agent_layers=2,
        dropout=0.2,
        k_neigh=8
    ):
        super().__init__()
        self.hist_len = hist_len
        self.num_agents = num_agents
        self.pred_len = pred_len
        self.k = k_neigh

        self.input_proj = nn.Linear(input_dim, model_dim)
        self.time_pos_embedding = nn.Parameter(torch.randn(1, hist_len, model_dim))
        time_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.time_encoder = nn.TransformerEncoder(time_layer, num_layers=time_layers)

        self.gat = GATConv(
            in_channels=model_dim,
            out_channels=model_dim // nhead,
            heads=nhead,
            concat=True,
            dropout=dropout,
        )

        self.agent_pos_embedding = nn.Parameter(torch.randn(1, num_agents, model_dim))
        agent_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.agent_encoder = nn.TransformerEncoder(agent_layer, num_layers=agent_layers)

        self.fc = nn.Linear(model_dim, pred_len * 2)

    def forward(self, data: Data):
        x, mask = data.x, data.mask
        if x.dim() == 3:
            total_agents, T, D = x.shape
            if total_agents == self.num_agents:
                x = x.unsqueeze(0)
                mask = mask.unsqueeze(0)
            else:
                B = mask.shape[0] // self.num_agents
                x = x.view(B, self.num_agents, T, D)
                mask = mask.view(B, self.num_agents)
        B, N, T, D = x.shape

        flat = x.view(B * N, T, D)
        h = self.input_proj(flat) + self.time_pos_embedding
        h = self.time_encoder(h)
        h_last = h[:, -1, :]
        agent_embed = h_last.view(B, N, -1)

        pos = x[..., :2].clone()
        last_pos = pos[:, :, -1, :]
        gat_out = []
        for b in range(B):
            emb_b = agent_embed[b]
            pos_b = last_pos[b]
            dist = torch.cdist(pos_b, pos_b)
            _, idx = torch.topk(dist, self.k + 1, largest=False)
            src = torch.arange(N, device=x.device).unsqueeze(1).repeat(1, self.k + 1).reshape(-1)
            dst = idx.reshape(-1)
            edge_index = torch.stack([src, dst], dim=0)
            valid = mask[b]
            valid_idx = valid.nonzero(as_tuple=False).view(-1)
            keep = torch.isin(src, valid_idx) & torch.isin(dst, valid_idx)
            edge_index = edge_index[:, keep]
            out_b = self.gat(emb_b, edge_index)
            gat_out.append(out_b)
        agent_embed = torch.stack(gat_out, dim=0)

        a = agent_embed + self.agent_pos_embedding
        pad_mask = ~mask
        a = self.agent_encoder(a, src_key_padding_mask=pad_mask)

        ego = a[:, 0, :]
        out = self.fc(ego)
        return out.view(B, self.pred_len, 2)


class SpatioTemporalTransformer(nn.Module):
    def __init__(
        self,
        input_dim=6,
        model_dim=128,
        hist_len=50,
        num_agents=50,
        pred_len=60,
        nhead=8,
        num_layers=4,
        dropout=0.2
    ):
        super().__init__()
        self.hist_len = hist_len
        self.num_agents = num_agents
        self.pred_len = pred_len

        self.input_proj = nn.Linear(input_dim, model_dim)
        self.time_pos_emb = nn.Parameter(torch.randn(1, hist_len, model_dim))
        self.agent_pos_emb = nn.Parameter(torch.randn(1, num_agents, model_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(model_dim, pred_len * 2)

    def forward(self, data: Data):
        x = data.x
        mask = data.mask
        if x.dim() == 3:
            total_agents, T, C = x.shape
            if total_agents == self.num_agents:
                x = x.unsqueeze(0)
                mask = mask.unsqueeze(0)
            else:
                B = mask.shape[0] // self.num_agents
                x = x.view(B, self.num_agents, T, C)
                mask = mask.view(B, self.num_agents)
        B, N, T, C = x.shape

        proj = self.input_proj(x)
        time_emb = self.time_pos_emb.unsqueeze(1)
        agent_emb = self.agent_pos_emb.unsqueeze(2)
        emb = proj + time_emb + agent_emb

        _, _, _, M = emb.shape
        seq = emb.reshape(B, N * T, M)
        pad_mask = ~mask
        pad_mask_flat = pad_mask.unsqueeze(-1).repeat(1, 1, T).reshape(B, N * T)

        enc = self.encoder(seq, src_key_padding_mask=pad_mask_flat)
        enc = enc.view(B, N, T, M)

        ego_hist = enc[:, 0, :, :]
        ego_repr = ego_hist.mean(dim=1)

        out = self.fc(ego_repr)
        return out.view(B, self.pred_len, 2)
