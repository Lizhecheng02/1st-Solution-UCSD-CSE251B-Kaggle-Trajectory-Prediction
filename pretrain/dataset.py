import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class TrajectoryDataset(Dataset):
    def __init__(self, data, is_train=True):
        self.data = data
        self.is_train = is_train
        self.num_scenes = len(data)

        self.input_steps = 19
        self.pred_steps = 30 if is_train else 0

        self.position_scaler, self.velocity_scaler = self._fit_scalers()

    def _fit_scalers(self):
        all_positions = []
        all_velocities = []

        for scene in self.data:
            p_in = scene["p_in"]
            v_in = scene["v_in"]

            mask = scene["car_mask"][:, None, None]
            p_masked = p_in * mask
            v_masked = v_in * mask

            all_positions.append(p_masked.reshape(-1, 2))
            all_velocities.append(v_masked.reshape(-1, 2))

        pos_data = np.concatenate(all_positions, axis=0)
        vel_data = np.concatenate(all_velocities, axis=0)

        pos_scaler = StandardScaler()
        vel_scaler = StandardScaler()

        pos_scaler.fit(pos_data[pos_data.any(axis=1)])
        vel_scaler.fit(vel_data[vel_data.any(axis=1)])

        return pos_scaler, vel_scaler

    def normalize(self, p, v):
        p_norm = self.position_scaler.transform(p.reshape(-1, 2)).reshape(p.shape)
        v_norm = self.velocity_scaler.transform(v.reshape(-1, 2)).reshape(v.shape)
        return p_norm, v_norm

    def denormalize_predictions(self, predictions):
        return self.position_scaler.inverse_transform(predictions.reshape(-1, 2)).reshape(predictions.shape)

    def __len__(self):
        return self.num_scenes

    def __getitem__(self, idx):
        scene = self.data[idx]
        p_in = scene["p_in"]
        v_in = scene["v_in"]
        p_out = scene.get("p_out", None)
        car_mask = scene["car_mask"]
        agent_id = scene["agent_id"]
        track_id = scene["track_id"]
        scene_idx = scene["scene_idx"]

        agent_index = track_id.index(agent_id)

        p_in_norm, v_in_norm = self.normalize(p_in, v_in)
        input_seq = np.concatenate([p_in_norm, v_in_norm], axis=-1)
        agent_input = input_seq[agent_index]

        if self.is_train:
            agent_output = p_out[agent_index]
            return {
                "ego_input": torch.FloatTensor(agent_input),
                "all_agents_input": torch.FloatTensor(input_seq),
                "valid_agents_mask": torch.BoolTensor(car_mask),
                "ego_future": torch.FloatTensor(agent_output)
            }
        else:
            return {
                "ego_input": torch.FloatTensor(agent_input),
                "all_agents_input": torch.FloatTensor(input_seq),
                "valid_agents_mask": torch.BoolTensor(car_mask),
                "scene_idx": scene_idx
            }
