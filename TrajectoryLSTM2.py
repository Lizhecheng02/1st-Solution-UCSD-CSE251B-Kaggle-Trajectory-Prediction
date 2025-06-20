# ================================================
# Created on Tue Apr 22 2025 12:25:59 PM
#
# The MIT License (MIT)
# Copyright (c) 2025
#
# Author: Zhecheng Li
# Institution: University of California, San Diego
# ================================================


from modules import PositionalEncoding, AgentTypeEmbedding, SocialLSTMEncoder, LSTMTrajectoryDecoder, AttentionDecoderWithMaskAndGating, InteractionEncoder
import torch
import torch.nn as nn


class TrajectoryLSTM2(nn.Module):
    def __init__(
        self,
        input_dim=5,
        d_model=64,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=256,
        dropout=0.1,
        max_len=50,
        pred_steps=60,
        num_agent_types=10,
        weights_initialization=False
    ):
        super(TrajectoryLSTM2, self).__init__()

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
        # self.attn_decoder = AttentionDecoderWithMaskAndGating(d_model, nhead, gating=True, distance_threshold=20.0)
        self.interaction_encoder = InteractionEncoder(d_model, nhead, gating=True, distance_threshold=30.0, min_agents=5, debug=False)

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

    # def forward(self, ego_input, all_agents_input, valid_agents_mask=None):
    #     batch_size = ego_input.shape[0]
    #     num_agents = all_agents_input.shape[1]
    #     seq_len = all_agents_input.shape[2]

    #     all_agents_flat = all_agents_input.view(batch_size * num_agents, seq_len, -1)
    #     agent_types = all_agents_flat[:, 0, 5].long()
    #     agent_features = all_agents_flat[:, :, :5]

    #     agent_embeddings = self.input_embedding(agent_features)
    #     type_embeddings = self.agent_type_embedding(agent_types).unsqueeze(1).expand(-1, seq_len, -1)
    #     agent_embeddings = agent_embeddings + type_embeddings

    #     agent_embeddings = self.positional_encoding(agent_embeddings)
    #     agent_embeddings = agent_embeddings.view(batch_size, num_agents, seq_len, -1)

    #     if valid_agents_mask is not None:
    #         mask = (~valid_agents_mask).unsqueeze(-1).unsqueeze(-1)
    #         agent_embeddings = agent_embeddings.masked_fill(mask, 0.0)

    #     agent_embeddings = agent_embeddings.view(batch_size * num_agents, seq_len, -1)
    #     encoded_features, _ = self.lstm_encoder(agent_embeddings)
    #     encoded_features = encoded_features[:, -1, :].view(batch_size, num_agents, -1)

    #     agent_positions = all_agents_input[:, :, -1, :2]
    #     last_pos = ego_input[:, -1, :2]
    #     decoder_input = self.decoder_input_embedding(last_pos).unsqueeze(1)

    #     predictions = []
    #     h_t = None
    #     c_t = None

    #     for _ in range(self.pred_steps):
    #         decoder_input = torch.clamp(decoder_input, -10, 10)

    #         context_input = self.attn_decoder(
    #             query=decoder_input,
    #             agent_contexts=encoded_features,
    #             ego_positions=last_pos,
    #             agent_positions=agent_positions
    #         )

    #         if h_t is None or c_t is None:
    #             output, (h_t, c_t) = self.trajectory_decoder(context_input)
    #         else:
    #             output, (h_t, c_t) = self.trajectory_decoder(context_input, (h_t, c_t))

    #         pred_pos = output[:, -1, :]
    #         predictions.append(pred_pos)
    #         decoder_input = self.decoder_input_embedding(pred_pos).unsqueeze(1)
    #         last_pos = pred_pos

    #     trajectory = torch.stack(predictions, dim=1)
    #     trajectory_flat = trajectory.reshape(batch_size, -1)
    #     refined_trajectory_flat = self.refinement_layer(trajectory_flat)
    #     refined_trajectory = refined_trajectory_flat.reshape(batch_size, self.pred_steps, 2)

    #     return refined_trajectory

    def forward(self, ego_input, all_agents_input, valid_agents_mask=None):
        B, N, T, _ = all_agents_input.shape

        agent_types = all_agents_input[:, :, 0, 5].long()
        agent_feats = all_agents_input[:, :, :, :5]

        x = self.input_embedding(agent_feats) + self.agent_type_embedding(agent_types).unsqueeze(2)
        x = self.positional_encoding(x.view(B * N, T, -1)).view(B, N, T, -1)

        enc_input = x.view(B * N, T, -1)
        enc_out, _ = self.lstm_encoder(enc_input)
        enc_feat = enc_out[:, -1, :].view(B, N, -1)

        ego_query = enc_feat[:, 0, :].unsqueeze(1)
        agent_positions = all_agents_input[:, :, -1, 0:2]
        ego_pos = ego_input[:, -1, 0:2]

        context_fused = self.interaction_encoder(
            query=ego_query,
            agent_contexts=enc_feat,
            ego_positions=ego_pos,
            agent_positions=agent_positions,
            valid_agents_mask=valid_agents_mask
        )

        decoder_input = self.decoder_input_embedding(ego_pos).unsqueeze(1)
        context_permuted = context_fused.permute(1, 0, 2).contiguous()
        h_0 = context_permuted.repeat(4, 1, 1)
        c_0 = torch.zeros_like(h_0)
        hidden = (h_0, c_0)

        predictions = []
        for _ in range(self.pred_steps):
            output, hidden = self.trajectory_decoder(decoder_input, hidden)
            pred_pos = output[:, -1, :]
            predictions.append(pred_pos)
            decoder_input = self.decoder_input_embedding(pred_pos).unsqueeze(1)

        trajectory = torch.stack(predictions, dim=1)

        trajectory_flat = trajectory.reshape(B, -1)
        refined = self.refinement_layer(trajectory_flat).reshape(B, self.pred_steps, 2)

        return refined

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
