import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class TrajectoryDataset(Dataset):
    def __init__(self, data, is_train=True):
        self.data = data
        self.is_train = is_train
        self.num_scenes = data.shape[0]

        self.input_steps = 50
        self.pred_steps = 60 if is_train else 0

        # Normalize data
        self.scalers = self._fit_scalers()

    def _fit_scalers(self):
        scalers = {}
        feature_names = ["position", "velocity", "heading"]
        feature_indices = [(0, 2), (2, 4), (4, 5)]

        for name, (start_idx, end_idx) in zip(feature_names, feature_indices):
            scaler = StandardScaler()
            flat_data = self.data[:, :, :, start_idx:end_idx].reshape(-1, end_idx - start_idx)
            non_zero_indices = np.any(flat_data != 0, axis=1)
            non_zero_data = flat_data[non_zero_indices]
            scaler.fit(non_zero_data)
            scalers[name] = scaler

        return scalers

    def normalize_data(self, data):
        normalized_data = data.copy()

        normalized_data[:, :, :, 0:2] = self.scalers["position"].transform(data[:, :, :, 0:2].reshape(-1, 2)).reshape(data.shape[0], data.shape[1], data.shape[2], 2)
        normalized_data[:, :, :, 2:4] = self.scalers["velocity"].transform(data[:, :, :, 2:4].reshape(-1, 2)).reshape(data.shape[0], data.shape[1], data.shape[2], 2)
        normalized_data[:, :, :, 4:5] = self.scalers["heading"].transform(data[:, :, :, 4:5].reshape(-1, 1)).reshape(data.shape[0], data.shape[1], data.shape[2], 1)

        return normalized_data

    def denormalize_predictions(self, predictions):
        return self.scalers["position"].inverse_transform(predictions.reshape(-1, 2)).reshape(predictions.shape[0], predictions.shape[1], 2)

    def __len__(self):
        return self.num_scenes

    def __getitem__(self, idx):
        scene_data = self.data[idx]

        scene_data = self.normalize_data(scene_data.reshape(1, *scene_data.shape))[0]

        input_seq = scene_data[:, :self.input_steps, :]

        ego_input = input_seq[0]

        valid_agents_mask = np.any(input_seq[:, :, :2] != 0, axis=(1, 2))

        valid_agents = input_seq[valid_agents_mask]

        if self.is_train:
            ego_future = scene_data[0, self.input_steps:self.input_steps + self.pred_steps, :2]
            return {
                "ego_input": torch.FloatTensor(ego_input),
                "all_agents_input": torch.FloatTensor(valid_agents),
                "valid_agents_mask": torch.BoolTensor(valid_agents_mask),
                "ego_future": torch.FloatTensor(ego_future)
            }
        else:
            return {
                "ego_input": torch.FloatTensor(ego_input),
                "all_agents_input": torch.FloatTensor(valid_agents),
                "valid_agents_mask": torch.BoolTensor(valid_agents_mask)
            }
