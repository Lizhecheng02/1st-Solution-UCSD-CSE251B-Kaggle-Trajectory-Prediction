import numpy as np
import torch
import os
import pickle
from tqdm import tqdm


def load_data():
    def load_pickle_folder(folder_path):
        scenes = []
        for file in sorted(os.listdir(folder_path)):
            if file.endswith(".pkl"):
                with open(os.path.join(folder_path, file), "rb") as f:
                    scene = pickle.load(f)
                    scenes.append(scene)
        return scenes

    train_data = load_pickle_folder("./data/train")
    test_data = load_pickle_folder("./data/val_in")

    print(f"Loaded {len(train_data)} training scenes")
    print(f"Loaded {len(test_data)} test scenes")

    return train_data, test_data


def evaluate_real_world_mse(model, dataloader, dataset, device):
    model.eval()
    total_loss = 0
    count = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Real-World MSE Evaluation"):
            ego_input = batch["ego_input"].to(device)
            all_agents_input = batch["all_agents_input"].to(device)
            valid_agents_mask = batch["valid_agents_mask"].to(device)
            ego_future = batch["ego_future"].cpu().numpy()

            predictions = model(ego_input, all_agents_input, valid_agents_mask)
            predictions_np = predictions.cpu().numpy()

            preds_denorm = dataset.denormalize_predictions(predictions_np)
            gt_denorm = dataset.denormalize_predictions(ego_future)

            mse = np.mean((preds_denorm - gt_denorm) ** 2)
            total_loss += mse
            count += 1

    return total_loss / count


def train_epoch(model, dataloader, optimizer, criterion, device, max_norm=1.0, model_type=None):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training"):
        ego_input = batch["ego_input"].to(device)
        all_agents_input = batch["all_agents_input"].to(device)
        valid_agents_mask = batch["valid_agents_mask"].to(device)
        ego_future = batch["ego_future"].to(device)

        optimizer.zero_grad()
        if model_type == "TrajectoryTransformer3":
            predictions = model(ego_input, all_agents_input, valid_agents_mask, ego_future=ego_future)
        else:
            predictions = model(ego_input, all_agents_input, valid_agents_mask)

        loss = criterion(predictions, ego_future)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device, model_type=None):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            ego_input = batch["ego_input"].to(device)
            all_agents_input = batch["all_agents_input"].to(device)
            valid_agents_mask = batch["valid_agents_mask"].to(device)
            ego_future = batch["ego_future"].to(device)

            if model_type == "TrajectoryTransformer3":
                predictions = model(ego_input, all_agents_input, valid_agents_mask, ego_future=None)
            else:
                predictions = model(ego_input, all_agents_input, valid_agents_mask)
            loss = criterion(predictions, ego_future)

            total_loss += loss.item()

    return total_loss / len(dataloader)
