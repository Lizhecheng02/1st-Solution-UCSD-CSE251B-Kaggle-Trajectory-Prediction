import torch
import numpy as np
from torch_geometric.data import Data, Dataset


class TrajectoryDatasetTrain(Dataset):
    def __init__(self, data, scale=10.0, augment=True):
        self.data = data
        self.scale = scale
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        scene = self.data[idx]
        hist = scene[:, :50, :].copy()
        future = torch.tensor(scene[0, 50:, :2].copy(), dtype=torch.float32)

        if self.augment:
            if np.random.rand() < 0.5:
                theta = np.random.uniform(-np.pi, np.pi)
                R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float32)
                hist[..., :2] = hist[..., :2] @ R
                hist[..., 2:4] = hist[..., 2:4] @ R
                future = future @ R
            if np.random.rand() < 0.5:
                hist[..., 0] *= -1
                hist[..., 2] *= -1
                future[:, 0] *= -1

        agent_sum = np.abs(hist).sum(axis=(1, 2))
        mask = agent_sum != 0

        origin = hist[0, 49, :2].copy()
        hist[..., :2] = hist[..., :2] - origin
        future = future - origin

        hist[..., :4] = hist[..., :4] / self.scale
        future = future / self.scale

        data_item = Data(
            x=torch.tensor(hist, dtype=torch.float32),
            mask=torch.tensor(mask, dtype=torch.bool),
            y=future.type(torch.float32),
            origin=torch.tensor(origin, dtype=torch.float32).unsqueeze(0),
            scale=torch.tensor(self.scale, dtype=torch.float32)
        )

        return data_item


class TrajectoryDatasetTest(Dataset):
    def __init__(self, data, scale=10.0):
        self.data = data
        self.scale = scale

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        scene = self.data[idx]
        hist = scene.copy()

        agent_sum = np.abs(hist).sum(axis=(1, 2))
        mask = agent_sum != 0

        origin = hist[0, 49, :2].copy()
        hist[..., :2] = hist[..., :2] - origin
        hist[..., :4] = hist[..., :4] / self.scale

        data_item = Data(
            x=torch.tensor(hist, dtype=torch.float32),
            mask=torch.tensor(mask, dtype=torch.bool),
            origin=torch.tensor(origin, dtype=torch.float32).unsqueeze(0),
            scale=torch.tensor(self.scale, dtype=torch.float32)
        )
        return data_item


class TrajectoryDatasetTrain2(Dataset):
    def __init__(self, data, scale=10.0, augment=True):
        self.data = data
        self.scale = scale
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        scene = self.data[idx]
        hist = scene[:, :50, :].copy()
        future = torch.tensor(scene[0, 50:, :2].copy(), dtype=torch.float32)

        if self.augment:
            if np.random.rand() < 0.5:
                theta = np.random.uniform(-np.pi, np.pi)
                R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float32)
                hist[..., :2] = hist[..., :2] @ R
                hist[..., 2:4] = hist[..., 2:4] @ R
                future = future @ R
            if np.random.rand() < 0.5:
                hist[..., 0] *= -1
                hist[..., 2] *= -1
                future[:, 0] *= -1

        agent_sum = np.abs(hist).sum(axis=(1, 2))
        mask = agent_sum != 0

        origin = hist[0, -1, :2].copy()
        hist[..., :2] -= origin
        future -= origin

        hist[..., :4] /= self.scale
        future /= self.scale

        N, T, D_orig = hist.shape
        vel = hist[..., 2:4]
        speed = np.linalg.norm(vel, axis=-1)
        accel = np.diff(vel, axis=1, prepend=np.zeros((N, 1, 2), dtype=vel.dtype))
        heading = hist[..., 4]
        yaw_rate = np.diff(heading, axis=1, prepend=np.zeros((N, 1), dtype=heading.dtype))

        speed_feat = speed[..., None]
        yaw_feat = yaw_rate[..., None]
        hist = np.concatenate([hist, speed_feat, accel, yaw_feat], axis=-1)

        data_item = Data(
            x=torch.tensor(hist, dtype=torch.float32),
            mask=torch.tensor(mask, dtype=torch.bool),
            y=future.type(torch.float32),
            origin=torch.tensor(origin, dtype=torch.float32).unsqueeze(0),
            scale=torch.tensor(self.scale, dtype=torch.float32)
        )

        return data_item


class TrajectoryDatasetTest2(Dataset):
    def __init__(self, data, scale=10.0):
        self.data = data
        self.scale = scale

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        scene = self.data[idx]
        hist = scene.copy()

        agent_sum = np.abs(hist).sum(axis=(1, 2))
        mask = agent_sum != 0

        origin = hist[0, 49, :2].copy()
        hist[..., :2] = hist[..., :2] - origin
        hist[..., :4] = hist[..., :4] / self.scale

        N, T, D_orig = hist.shape
        vel = hist[..., 2:4]
        speed = np.linalg.norm(vel, axis=-1)
        accel = np.diff(vel, axis=1, prepend=np.zeros((N, 1, 2), dtype=vel.dtype))
        heading = hist[..., 4]
        yaw_rate = np.diff(heading, axis=1, prepend=np.zeros((N, 1), dtype=heading.dtype))

        speed_feat = speed[..., None]
        yaw_feat = yaw_rate[..., None]
        hist = np.concatenate([hist, speed_feat, accel, yaw_feat], axis=-1)

        data_item = Data(
            x=torch.tensor(hist, dtype=torch.float32),
            mask=torch.tensor(mask, dtype=torch.bool),
            origin=torch.tensor(origin, dtype=torch.float32).unsqueeze(0),
            scale=torch.tensor(self.scale, dtype=torch.float32)
        )
        return data_item


class TrajectoryDatasetTrain3(Dataset):
    def __init__(self, data, scale=10.0, augment=True):
        self.data = data
        self.scale = scale
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        scene = self.data[idx]
        hist = scene[:, :50, :].copy()
        future = torch.tensor(scene[0, 50:, :2].copy(), dtype=torch.float32)

        if self.augment:
            if np.random.rand() < 0.5:
                theta = np.random.uniform(-np.pi, np.pi)
                R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float32)
                hist[..., :2] = hist[..., :2] @ R
                hist[..., 2:4] = hist[..., 2:4] @ R
                future = future @ R
            if np.random.rand() < 0.5:
                hist[..., 0] *= -1
                hist[..., 2] *= -1
                future[:, 0] *= -1

        agent_sum = np.abs(hist).sum(axis=(1, 2))
        mask = agent_sum != 0

        origin = hist[0, -1, :2].copy()
        hist[..., :2] -= origin
        future -= origin

        hist[..., :4] /= self.scale
        future /= self.scale

        N, T, D_orig = hist.shape
        vel = hist[..., 2:4]
        speed = np.linalg.norm(vel, axis=-1)
        accel = np.diff(vel, axis=1, prepend=np.zeros((N, 1, 2), dtype=vel.dtype))
        heading = hist[..., 4]
        yaw_rate = np.diff(heading, axis=1, prepend=np.zeros((N, 1), dtype=heading.dtype))

        speed_feat = speed[..., None]
        yaw_feat = yaw_rate[..., None]

        def compute_shift_diff(arr, k):
            pad = np.zeros((N, k) + arr.shape[2:], dtype=arr.dtype)
            shifted = np.concatenate([pad, arr[:, :-k]], axis=1)
            return arr - shifted

        vel_diff2 = compute_shift_diff(vel, 2)
        vel_diff3 = compute_shift_diff(vel, 3)
        vel_diff4 = compute_shift_diff(vel, 4)
        vel_diff5 = compute_shift_diff(vel, 5)
        vel_diff10 = compute_shift_diff(vel, 10)
        vel_diff15 = compute_shift_diff(vel, 15)
        vel_diff20 = compute_shift_diff(vel, 20)

        yaw_rate2 = compute_shift_diff(heading[..., None], 2)[..., 0]
        yaw_rate3 = compute_shift_diff(heading[..., None], 3)[..., 0]
        yaw_rate4 = compute_shift_diff(heading[..., None], 4)[..., 0]
        yaw_rate5 = compute_shift_diff(heading[..., None], 5)[..., 0]
        yaw_rate10 = compute_shift_diff(heading[..., None], 10)[..., 0]
        yaw_rate15 = compute_shift_diff(heading[..., None], 15)[..., 0]
        yaw_rate20 = compute_shift_diff(heading[..., None], 20)[..., 0]

        yaw2_feat = yaw_rate2[..., None]
        yaw3_feat = yaw_rate3[..., None]
        yaw4_feat = yaw_rate4[..., None]
        yaw5_feat = yaw_rate5[..., None]
        yaw10_feat = yaw_rate10[..., None]
        yaw15_feat = yaw_rate15[..., None]
        yaw20_feat = yaw_rate20[..., None]

        hist = np.concatenate([
            hist,
            speed_feat,
            accel,
            yaw_feat,
            vel_diff2,
            vel_diff3,
            vel_diff4,
            vel_diff5,
            vel_diff10,
            vel_diff15,
            vel_diff20,
            yaw2_feat,
            yaw3_feat,
            yaw4_feat,
            yaw5_feat,
            yaw10_feat,
            yaw15_feat,
            yaw20_feat
        ], axis=-1)

        data_item = Data(
            x=torch.tensor(hist, dtype=torch.float32),
            mask=torch.tensor(mask, dtype=torch.bool),
            y=future.type(torch.float32),
            origin=torch.tensor(origin, dtype=torch.float32).unsqueeze(0),
            scale=torch.tensor(self.scale, dtype=torch.float32)
        )

        return data_item


class TrajectoryDatasetTest3(Dataset):
    def __init__(self, data, scale=10.0):
        self.data = data
        self.scale = scale

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        scene = self.data[idx]
        hist = scene.copy()

        agent_sum = np.abs(hist).sum(axis=(1, 2))
        mask = agent_sum != 0

        origin = hist[0, 49, :2].copy()
        hist[..., :2] = hist[..., :2] - origin
        hist[..., :4] = hist[..., :4] / self.scale

        N, T, D_orig = hist.shape
        vel = hist[..., 2:4]
        speed = np.linalg.norm(vel, axis=-1)
        accel = np.diff(vel, axis=1, prepend=np.zeros((N, 1, 2), dtype=vel.dtype))
        heading = hist[..., 4]
        yaw_rate = np.diff(heading, axis=1, prepend=np.zeros((N, 1), dtype=heading.dtype))

        speed_feat = speed[..., None]
        yaw_feat = yaw_rate[..., None]

        def compute_shift_diff(arr, k):
            pad = np.zeros((N, k) + arr.shape[2:], dtype=arr.dtype)
            shifted = np.concatenate([pad, arr[:, :-k]], axis=1)
            return arr - shifted

        vel_diff2 = compute_shift_diff(vel, 2)
        vel_diff3 = compute_shift_diff(vel, 3)
        vel_diff4 = compute_shift_diff(vel, 4)
        vel_diff5 = compute_shift_diff(vel, 5)
        vel_diff10 = compute_shift_diff(vel, 10)
        vel_diff15 = compute_shift_diff(vel, 15)
        vel_diff20 = compute_shift_diff(vel, 20)

        yaw_rate2 = compute_shift_diff(heading[..., None], 2)[..., 0]
        yaw_rate3 = compute_shift_diff(heading[..., None], 3)[..., 0]
        yaw_rate4 = compute_shift_diff(heading[..., None], 4)[..., 0]
        yaw_rate5 = compute_shift_diff(heading[..., None], 5)[..., 0]
        yaw_rate10 = compute_shift_diff(heading[..., None], 10)[..., 0]
        yaw_rate15 = compute_shift_diff(heading[..., None], 15)[..., 0]
        yaw_rate20 = compute_shift_diff(heading[..., None], 20)[..., 0]

        yaw2_feat = yaw_rate2[..., None]
        yaw3_feat = yaw_rate3[..., None]
        yaw4_feat = yaw_rate4[..., None]
        yaw5_feat = yaw_rate5[..., None]
        yaw10_feat = yaw_rate10[..., None]
        yaw15_feat = yaw_rate15[..., None]
        yaw20_feat = yaw_rate20[..., None]

        hist = np.concatenate([
            hist,
            speed_feat,
            accel,
            yaw_feat,
            vel_diff2,
            vel_diff3,
            vel_diff4,
            vel_diff5,
            vel_diff10,
            vel_diff15,
            vel_diff20,
            yaw2_feat,
            yaw3_feat,
            yaw4_feat,
            yaw5_feat,
            yaw10_feat,
            yaw15_feat,
            yaw20_feat
        ], axis=-1)

        data_item = Data(
            x=torch.tensor(hist, dtype=torch.float32),
            mask=torch.tensor(mask, dtype=torch.bool),
            origin=torch.tensor(origin, dtype=torch.float32).unsqueeze(0),
            scale=torch.tensor(self.scale, dtype=torch.float32)
        )
        return data_item
