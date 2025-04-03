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


class TrajectoryTransformer1(nn.Module):
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
        num_agent_types=10
    ):
        super(TrajectoryTransformer2, self).__init__()

        self.d_model = d_model
        self.pred_steps = pred_steps

        self.input_embedding = nn.Linear(input_dim, d_model)
        self.agent_type_embedding = AgentTypeEmbedding(num_agent_types, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        encoder_layer = SocialTransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        self.trajectory_decoder = LSTMTrajectoryDecoder(
            input_dim=d_model,
            hidden_dim=dim_feedforward,
            output_dim=2,
            num_layers=num_decoder_layers,
            dropout=dropout
        )

        # Initial decoder input
        self.decoder_input_embedding = nn.Linear(2, d_model)

        self.refinement_layer = nn.Sequential(
            nn.Linear(2 * pred_steps, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 2 * pred_steps)
        )

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

        social_encodings = []
        for t in range(seq_len):
            timestep_embeddings = agent_embeddings[:, :, t, :]

            mask = ~valid_agents_mask if valid_agents_mask is not None else None

            timestep_encoded = self.transformer_encoder(
                timestep_embeddings,
                src_key_padding_mask=mask
            )

            social_encodings.append(timestep_encoded)

        social_encodings = torch.stack(social_encodings, dim=2)

        # ego_encodings = social_encodings[:, 0, :, :]

        last_pos = ego_input[:, -1, :2]
        decoder_input = self.decoder_input_embedding(last_pos).unsqueeze(1)

        predictions = []
        hidden = None

        for t in range(self.pred_steps):
            output, hidden = self.trajectory_decoder(decoder_input, hidden)

            pred_pos = output[:, -1, :]
            predictions.append(pred_pos)

            decoder_input = self.decoder_input_embedding(pred_pos).unsqueeze(1)

        trajectory = torch.stack(predictions, dim=1)

        trajectory_flat = trajectory.reshape(batch_size, -1)
        refined_trajectory_flat = self.refinement_layer(trajectory_flat)
        refined_trajectory = refined_trajectory_flat.reshape(batch_size, self.pred_steps, 2)

        return refined_trajectory


class TrajectoryTransformer2(nn.Module):
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
        num_agent_types=10
    ):
        super(TrajectoryTransformer2, self).__init__()

        self.d_model = d_model
        self.pred_steps = pred_steps

        self.input_embedding = nn.Linear(input_dim, d_model)
        self.agent_type_embedding = AgentTypeEmbedding(num_agent_types, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        encoder_layer = SocialTransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        self.trajectory_decoder = TransformerTrajectoryDecoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=pred_steps
        )

        self.decoder_input_embedding = nn.Linear(2, d_model)

        self.refinement_layer = nn.Sequential(
            nn.Linear(2 * pred_steps, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 2 * pred_steps)
        )

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

        social_encodings = []
        for t in range(seq_len):
            timestep_embeddings = agent_embeddings[:, :, t, :]
            mask = ~valid_agents_mask if valid_agents_mask is not None else None
            timestep_encoded = self.transformer_encoder(
                timestep_embeddings,
                src_key_padding_mask=mask
            )
            social_encodings.append(timestep_encoded)

        social_encodings = torch.stack(social_encodings, dim=2)
        ego_encodings = social_encodings[:, 0, :, :]

        last_pos = ego_input[:, -1, :2]
        first_decoder_input = self.decoder_input_embedding(last_pos).unsqueeze(1)
        decoder_inputs = torch.zeros(batch_size, self.pred_steps, self.d_model, device=ego_input.device)
        decoder_inputs[:, 0:1, :] = first_decoder_input

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(self.pred_steps).to(ego_input.device)
        decoder_outputs = self.trajectory_decoder(decoder_inputs, ego_encodings, tgt_mask=tgt_mask)

        trajectory = decoder_outputs
        trajectory_flat = trajectory.reshape(batch_size, -1)
        refined_trajectory_flat = self.refinement_layer(trajectory_flat)
        refined_trajectory = refined_trajectory_flat.reshape(batch_size, self.pred_steps, 2)

        return refined_trajectory


class TrajectoryTransformer3(nn.Module):
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
        num_agent_types=10
    ):
        super(TrajectoryTransformer2, self).__init__()

        self.d_model = d_model
        self.pred_steps = pred_steps

        self.input_embedding = nn.Linear(input_dim, d_model)
        self.agent_type_embedding = AgentTypeEmbedding(num_agent_types, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        encoder_layer = SocialTransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        self.trajectory_decoder = TransformerTrajectoryDecoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=pred_steps
        )

        self.decoder_input_embedding = nn.Linear(2, d_model)

        self.refinement_layer = nn.Sequential(
            nn.Linear(2 * pred_steps, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 2 * pred_steps)
        )

    def forward(self, ego_input, all_agents_input, valid_agents_mask=None, ego_future=None):
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

        social_encodings = []
        for t in range(seq_len):
            timestep_embeddings = agent_embeddings[:, :, t, :]
            mask = ~valid_agents_mask if valid_agents_mask is not None else None
            timestep_encoded = self.transformer_encoder(
                timestep_embeddings,
                src_key_padding_mask=mask
            )
            social_encodings.append(timestep_encoded)

        social_encodings = torch.stack(social_encodings, dim=2)
        ego_encodings = social_encodings[:, 0, :, :]

        if ego_future is not None:
            decoder_inputs = self.decoder_input_embedding(ego_future[:, :-1])
            last_pos = ego_input[:, -1, :2]
            start_token = self.decoder_input_embedding(last_pos).unsqueeze(1)
            decoder_inputs = torch.cat([start_token, decoder_inputs], dim=1)

            tgt_mask = nn.Transformer.generate_square_subsequent_mask(self.pred_steps).to(ego_input.device)
            decoder_outputs = self.trajectory_decoder(decoder_inputs, ego_encodings, tgt_mask=tgt_mask)
            trajectory = decoder_outputs
        else:
            predictions = []
            decoder_input = self.decoder_input_embedding(ego_input[:, -1, :2]).unsqueeze(1)
            for t in range(self.pred_steps):
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(t + 1).to(ego_input.device)
                decoder_output = self.trajectory_decoder(decoder_input, ego_encodings, tgt_mask=tgt_mask)
                pred_pos = decoder_output[:, -1, :]
                predictions.append(pred_pos)
                next_input = self.decoder_input_embedding(pred_pos).unsqueeze(1)
                decoder_input = torch.cat([decoder_input, next_input], dim=1)
            trajectory = torch.stack(predictions, dim=1)

        trajectory_flat = trajectory.reshape(batch_size, -1)
        refined_trajectory_flat = self.refinement_layer(trajectory_flat)
        refined_trajectory = refined_trajectory_flat.reshape(batch_size, self.pred_steps, 2)
        return refined_trajectory
