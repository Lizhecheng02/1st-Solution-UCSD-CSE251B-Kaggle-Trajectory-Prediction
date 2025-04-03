import csv
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader


def load_data():
    train_file = np.load("./data/train.npz")
    train_data = train_file["data"]
    print("train_data shape", train_data.shape)

    test_file = np.load("./data/test_input.npz")
    test_data = test_file["data"]
    print("test_data shape", test_data.shape)

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


def train_epoch(model, dataloader, optimizer, criterion, device, model_type=None):
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
                predictions = model(ego_input, all_agents_input, valid_agents_mask, ego_future=ego_future)
            else:
                predictions = model(ego_input, all_agents_input, valid_agents_mask)
            loss = criterion(predictions, ego_future)

            total_loss += loss.item()

    return total_loss / len(dataloader)


def predict(model, dataset, device):
    model.eval()
    predictions = []

    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            ego_input = batch["ego_input"].to(device)
            all_agents_input = batch["all_agents_input"].to(device)
            valid_agents_mask = batch["valid_agents_mask"].to(device)

            batch_predictions = model(ego_input, all_agents_input, valid_agents_mask)

            batch_predictions_np = batch_predictions.cpu().numpy()
            denormalized_predictions = dataset.denormalize_predictions(batch_predictions_np)

            predictions.append(denormalized_predictions)

    return np.concatenate(predictions, axis=0)


def create_submission(predictions, output_file="submission.csv"):
    num_scenes, num_agents, pred_steps, dimensions = predictions.shape
    assert num_agents == 1, "Expected exactly 1 ego agent"
    assert dimensions == 2, "Expected 2D predictions (x, y)"

    with open(output_file, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["ID", "x", "y"])

        for scene_id in range(num_scenes):
            for t in range(pred_steps):
                x, y = predictions[scene_id, 0, t]
                row_id = scene_id * pred_steps + t
                writer.writerow([row_id, x, y])

    print(f"Submission file created: {output_file}")
