# ================================================
# Created on Sat Apr 19 2025 8:00:12 PM
#
# The MIT License (MIT)
# Copyright (c) 2025
#
# Author: Zhecheng Li
# Institution: University of California, San Diego
# ================================================


import csv
import numpy as np
import torch
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
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


def compute_ade_fde(pred, gt, alpha=2.0):
    displacement = torch.norm(pred - gt, dim=-1)
    ade = displacement.mean()
    fde = displacement[:, -1].mean()
    loss = ade + alpha * fde
    return loss


def compute_weighted_loss(pred, gt, mode="linear"):
    _, T, _ = pred.shape
    displacement = torch.norm(pred - gt, dim=-1)

    if mode == "linear":
        weights = torch.linspace(1.0, 2.0, T, device=pred.device)
    elif mode == "sqrt":
        weights = torch.sqrt(torch.linspace(1.0, T, T, device=pred.device))
    elif mode == "exp":
        weights = torch.exp(torch.linspace(0.0, 1.0, T, device=pred.device))
    else:
        weights = torch.ones(T, device=pred.device)

    weights = weights / weights.sum() * T

    weighted_loss = (displacement * weights).mean()
    return weighted_loss


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

        # loss = compute_weighted_loss(predictions, ego_future, mode="linear")
        # loss = compute_ade_fde(predictions, ego_future, alpha=2.0)
        loss = criterion(predictions, ego_future)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            ego_input = batch["ego_input"].to(device)
            all_agents_input = batch["all_agents_input"].to(device)
            valid_agents_mask = batch["valid_agents_mask"].to(device)
            ego_future = batch["ego_future"].to(device)

            predictions = model(ego_input, all_agents_input, valid_agents_mask)

            # loss = compute_weighted_loss(predictions, ego_future, mode="linear")
            # loss = compute_ade_fde(predictions, ego_future, alpha=2.0)
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
            ego_start_pos = batch["ego_start_pos"].cpu().numpy()

            batch_predictions = model(ego_input, all_agents_input, valid_agents_mask)
            batch_predictions_np = batch_predictions.cpu().numpy()

            denormalized_predictions = batch_predictions_np + ego_start_pos[:, np.newaxis, :]
            denormalized_predictions = denormalized_predictions[:, np.newaxis, :, :]

            predictions.append(denormalized_predictions)

    return np.concatenate(predictions, axis=0)


def create_submission(predictions, output_file="submission.csv"):
    num_scenes, num_agents, pred_steps, dimensions = predictions.shape

    assert num_agents == 1, "Expected exactly 1 ego agent"
    assert dimensions == 2, "Expected 2D predictions (x, y)"

    with open(output_file, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["index", "x", "y"])

        for scene_id in range(num_scenes):
            for t in range(pred_steps):
                x, y = predictions[scene_id, 0, t]
                row_id = scene_id * pred_steps + t
                writer.writerow([row_id, x, y])

    print(f"Submission file created: {output_file}")


def ensemble_submissions(submission_dir, output_path):
    if isinstance(submission_dir, str):
        submission_dir = [submission_dir]

    csv_files = []
    for dir in submission_dir:
        csv_files = glob.glob(os.path.join(dir, "**", "*.csv"), recursive=True)

    if not csv_files:
        print("No CSV files found in the provided directory/directories.")
        return

    merged_df = None

    for _, file in enumerate(csv_files):
        df = pd.read_csv(file)
        df = df.set_index("index")
        if merged_df is None:
            merged_df = df
        else:
            merged_df += df

    ensembled_df = merged_df / len(csv_files)
    ensembled_df = ensembled_df.reset_index()

    ensembled_df.to_csv(output_path, index=False)
    print(f"Ensembled submission saved to {output_path}")


def plot_loss_curve(train_losses, val_losses, fold, total_folds, model_type):
    os.makedirs("figures", exist_ok=True)
    mark_interval = max(1, len(train_losses) // 10)
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, marker="^", markevery=mark_interval, color="tab:blue", label="Train Loss")
    plt.plot(val_losses, marker="s", markevery=mark_interval, color="tab:orange", label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_type} Training and Validation Loss (Fold {fold}/{total_folds})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"figures/{model_type}-Fold-{fold}-Loss.pdf", format="pdf", dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close()
