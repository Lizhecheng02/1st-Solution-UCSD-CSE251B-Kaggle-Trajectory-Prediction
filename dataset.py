import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    def __init__(self, data, is_train=True):
        self.data = data
        self.is_train = is_train
        self.num_scenes = data.shape[0]

        self.input_steps = 50
        self.pred_steps = 60 if is_train else 0

    def center_ego_position(self, data):
        centered_data = data.copy()
        ego_start_pos = data[:, 0, 0, 0:2]

        centered_data[:, :, :, 0:2] -= ego_start_pos[:, None, None, :]

        return centered_data, ego_start_pos

    def denormalize_predictions(self, predictions):
        return predictions

    def __len__(self):
        return self.num_scenes

    def __getitem__(self, idx):
        scene_data = self.data[idx]
        scene_data, ego_start = self.center_ego_position(scene_data[np.newaxis, ...])
        scene_data = scene_data[0]

        input_seq = scene_data[:, :self.input_steps, :]
        ego_input = input_seq[0]

        valid_agents_mask = np.any(input_seq[:, :, 0:2] != 0, axis=(1, 2))
        valid_agents = input_seq[valid_agents_mask]

        if self.is_train:
            ego_future = scene_data[0, self.input_steps:self.input_steps + self.pred_steps, 0:2]

            return {
                "ego_input": torch.FloatTensor(ego_input),
                "all_agents_input": torch.FloatTensor(valid_agents),
                "valid_agents_mask": torch.BoolTensor(valid_agents_mask),
                "ego_future": torch.FloatTensor(ego_future),
                "ego_start_pos": torch.FloatTensor(ego_start[0])
            }
        else:
            return {
                "ego_input": torch.FloatTensor(ego_input),
                "all_agents_input": torch.FloatTensor(valid_agents),
                "valid_agents_mask": torch.BoolTensor(valid_agents_mask),
                "ego_start_pos": torch.FloatTensor(ego_start[0])
            }
