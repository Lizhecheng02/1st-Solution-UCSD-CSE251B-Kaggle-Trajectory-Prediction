# ================================================
# Created on Sat Apr 19 2025 5:15:53 PM
#
# The MIT License (MIT)
# Copyright (c) 2025
#
# Author: Zhecheng Li
# Institution: University of California, San Diego
# ================================================


from modules import PositionalEncoding, AgentTypeEmbedding, SocialLSTMEncoder, LSTMTrajectoryDecoder
import torch
import torch.nn as nn


class TrajectoryLSTM(nn.Module):
    def __init__(
        self,
        input_dim=5,
        d_model=128,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=512,
        dropout=0.1,
        max_len=50,
        pred_steps=60,
        num_agent_types=10,
        weights_initialization=False
    ):
        super(TrajectoryLSTM, self).__init__()

        self.d_model = d_model
        self.pred_steps = pred_steps

        self.input_embedding = nn.Linear(input_dim, d_model)
        self.agent_type_embedding = AgentTypeEmbedding(num_agent_types, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        self.lstm_encoder = SocialLSTMEncoder(
            input_dim=d_model,
            hidden_dim=d_model,
            num_layers=num_encoder_layers,
            dropout=dropout
        )

        self.trajectory_decoder = LSTMTrajectoryDecoder(
            input_dim=d_model,
            hidden_dim=d_model,
            output_dim=2,
            num_layers=num_decoder_layers,
            dropout=dropout
        )

        self.decoder_input_embedding = nn.Linear(2, d_model)

        self.refinement_layer = nn.Sequential(
            nn.Linear(2 * pred_steps, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 2 * pred_steps)
        )

        self.refinement_layer2 = nn.Sequential(
            nn.Linear(2 * pred_steps, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 2 * pred_steps)
        )

        self.refinement_layer3 = nn.Sequential(
            nn.Linear(2 * pred_steps, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2 * pred_steps)
        )

        if weights_initialization:
            self.apply(self._init_weights)
            print("Weights initialized")

    def forward(self, ego_input, all_agents_input, valid_agents_mask=None):
        batch_size = ego_input.shape[0]
        num_agents = all_agents_input.shape[1]
        seq_len = all_agents_input.shape[2]

        all_agents_flat = all_agents_input.view(batch_size * num_agents, seq_len, -1)

        agent_types = all_agents_flat[:, 0, 5].long()
        agent_features = all_agents_flat[:, :, :5]

        agent_embeddings = self.input_embedding(agent_features)
        type_embeddings = self.agent_type_embedding(agent_types).unsqueeze(1).expand(-1, seq_len, -1)
        agent_embeddings = agent_embeddings + type_embeddings

        agent_embeddings = self.positional_encoding(agent_embeddings)

        agent_embeddings = agent_embeddings.view(batch_size, num_agents, seq_len, -1)

        if valid_agents_mask is not None:
            mask = (~valid_agents_mask).unsqueeze(-1).unsqueeze(-1)
            agent_embeddings = agent_embeddings.masked_fill(mask, 0.0)

        ego_embeddings = agent_embeddings[:, 0, :, :]

        _, (h_n, c_n) = self.lstm_encoder(ego_embeddings)

        last_pos = ego_input[:, -1, :2]
        decoder_input = self.decoder_input_embedding(last_pos).unsqueeze(1)

        predictions = []
        hidden = (h_n, c_n)

        for _ in range(self.pred_steps):
            output, hidden = self.trajectory_decoder(decoder_input, hidden)
            pred_pos = output[:, -1, :]
            predictions.append(pred_pos)
            decoder_input = self.decoder_input_embedding(pred_pos).unsqueeze(1)

        trajectory = torch.stack(predictions, dim=1)

        trajectory_flat = trajectory.reshape(batch_size, -1)
        refined_trajectory_flat = self.refinement_layer(trajectory_flat)
        refined_trajectory = refined_trajectory_flat.reshape(batch_size, self.pred_steps, 2)

        return refined_trajectory

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if "weight" in name:
                    nn.init.kaiming_uniform_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)
