import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
import time
import random
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import json
from torch_geometric.data import DataLoader, Batch
from dataset import TrajectoryDatasetTrain, TrajectoryDatasetTest, TrajectoryDatasetTrain2, TrajectoryDatasetTest2, TrajectoryDatasetTrain3, TrajectoryDatasetTest3
from init import seed_everything, get_device
from models import LSTMNet, TransformerNet, InteractionTransformer, InteractionTransformerWithGAT, SpatioTemporalTransformer
from tqdm import tqdm
from utils import plot_trajectory, plot_loss_curve, compute_ade_fde, compute_weighted_loss
from sklearn.model_selection import KFold
warnings.filterwarnings("ignore")


def train(args):
    seed_everything()
    device = get_device()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    SAVE_DIR = args.save_dir + f"/Fold{args.fold}/{args.model}/{int(time.time())}"
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Saving models to {SAVE_DIR}")

    train_npz = np.load("../data/train.npz")
    train_data = train_npz["data"]
    test_npz = np.load("../data/test_input.npz")
    test_data = test_npz["data"]

    print(train_data.shape, test_data.shape)

    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=777)

    folds = list(kf.split(train_data))
    train_idx, val_idx = folds[args.fold]

    if args.data_version == "v1":
        train_dataset = TrajectoryDatasetTrain([train_data[i] for i in train_idx], scale=args.scale, augment=True)
        val_dataset = TrajectoryDatasetTrain([train_data[i] for i in val_idx], scale=args.scale, augment=False)
    elif args.data_version == "v2":
        train_dataset = TrajectoryDatasetTrain2([train_data[i] for i in train_idx], scale=args.scale, augment=True)
        val_dataset = TrajectoryDatasetTrain2([train_data[i] for i in val_idx], scale=args.scale, augment=False)
    elif args.data_version == "v3":
        train_dataset = TrajectoryDatasetTrain3([train_data[i] for i in train_idx], scale=args.scale, augment=True)
        val_dataset = TrajectoryDatasetTrain3([train_data[i] for i in val_idx], scale=args.scale, augment=False)
    else:
        raise ValueError(f"Data version {args.data_version} Not Found")

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: Batch.from_data_list(x))
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: Batch.from_data_list(x))

    if args.model == "LSTMNet":
        model = LSTMNet(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            output_dim=args.output_dim
        ).to(device)
    elif args.model == "TransformerNet":
        model = TransformerNet(
            input_dim=args.input_dim,
            model_dim=args.hidden_dim,
            output_dim=args.output_dim,
            nhead=args.nhead,
            num_layers=args.num_layers
        ).to(device)
    elif args.model == "InteractionTransformer":
        model = InteractionTransformer(
            input_dim=args.input_dim,
            model_dim=args.hidden_dim,
            hist_len=50,
            num_agents=50,
            pred_len=60,
            nhead=args.nhead,
            time_layers=args.time_layers,
            agent_layers=args.agent_layers,
            dropout=args.dropout
        ).to(device)
    elif args.model == "InteractionTransformerWithGAT":
        model = InteractionTransformerWithGAT(
            input_dim=args.input_dim,
            model_dim=args.hidden_dim,
            hist_len=50,
            num_agents=50,
            pred_len=60,
            nhead=args.nhead,
            time_layers=args.time_layers,
            agent_layers=args.agent_layers,
            dropout=args.dropout,
            k_neigh=args.k_neigh
        ).to(device)
    elif args.model == "SpatioTemporalTransformer":
        model = SpatioTemporalTransformer(
            input_dim=args.input_dim,
            model_dim=args.hidden_dim,
            hist_len=50,
            num_agents=50,
            pred_len=60,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dropout=args.dropout
        ).to(device)
    else:
        raise ValueError(f"Model {args.model} Not Found")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params}")

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    best_val_mse_loss = float("inf")
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []
    val_maes = []
    val_mses = []

    for epoch in tqdm(range(args.num_epochs), desc="Training"):
        model.train()
        train_loss = 0
        for batch in train_dataloader:
            batch = batch.to(device)
            pred = model(batch)
            y = batch.y.view(batch.num_graphs, 60, 2)
            loss = criterion(pred, y)
            # loss = compute_weighted_loss(pred, y, mode="sqrt")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        val_mae = 0
        val_mse = 0

        with torch.no_grad():
            for batch in val_dataloader:
                batch = batch.to(device)
                pred = model(batch)
                y = batch.y.view(batch.num_graphs, 60, 2)
                val_loss += criterion(pred, y).item()

                pred = pred * batch.scale.view(-1, 1, 1) + batch.origin.unsqueeze(1)
                y = y * batch.scale.view(-1, 1, 1) + batch.origin.unsqueeze(1)
                val_mae += nn.L1Loss()(pred, y).item()
                val_mse += nn.MSELoss()(pred, y).item()

        train_loss /= len(train_dataloader)
        val_loss /= len(val_dataloader)
        val_mae /= len(val_dataloader)
        val_mse /= len(val_dataloader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_maes.append(val_mae)
        val_mses.append(val_mse)

        scheduler.step()

        tqdm.write(f"Epoch {epoch:03d} | Learning rate {optimizer.param_groups[0]['lr']:.6f} | train normalized MSE {train_loss:8.4f} | val normalized MSE {val_loss:8.4f}, | val MAE {val_mae:8.4f} | val MSE {val_mse:8.4f}")
        if val_mse < best_val_mse_loss:
            best_val_mse_loss = val_mse
            torch.save(model.state_dict(), f"{SAVE_DIR}/best_model.pt")
            print(f"The best model saved! - epoch {epoch}")

    plot_loss_curve(train_losses, val_losses, val_maes, val_mses, args.fold, args.n_folds, args.model)

    result_dict = vars(args).copy()
    result_dict["best_val_mse_loss"] = best_val_mse_loss

    json_path = os.path.join(SAVE_DIR, "config.json")
    with open(json_path, "w") as f:
        json.dump(result_dict, f, indent=4)

    model.load_state_dict(torch.load(f"{SAVE_DIR}/best_model.pt"))
    model.eval()

    random_indices = random.sample(range(len(val_dataset)), 4)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for i, idx in enumerate(random_indices):
        batch = val_dataset[idx]
        batch = batch.to(device)
        pred = model(batch)
        gt = torch.stack(torch.split(batch.y, 60, dim=0), dim=0)

        pred = pred * batch.scale.view(-1, 1, 1) + batch.origin.unsqueeze(1)
        gt = torch.stack(torch.split(batch.y, 60, dim=0), dim=0) * batch.scale.view(-1, 1, 1) + batch.origin.unsqueeze(1)

        pred = pred.detach().cpu().numpy()
        gt = gt.detach().cpu().numpy()

        plot_trajectory(axes[i], pred, gt, title=f"Sample {idx}")

    fig.savefig(f"{SAVE_DIR}/trajectory_example.png", dpi=300, bbox_inches="tight", pad_inches=0.05)

    model = model.to(device)
    model.eval()

    if args.data_version == "v1":
        test_dataset = TrajectoryDatasetTest(test_data, scale=args.scale)
    elif args.data_version == "v2":
        test_dataset = TrajectoryDatasetTest2(test_data, scale=args.scale)
    elif args.data_version == "v3":
        test_dataset = TrajectoryDatasetTest3(test_data, scale=args.scale)
    else:
        raise ValueError(f"Data version {args.data_version} Not Found")

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda xs: Batch.from_data_list(xs))

    pred_list = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred_norm = model(batch)
            pred = pred_norm * batch.scale.view(-1, 1, 1) + batch.origin.unsqueeze(1)
            pred_list.append(pred.cpu().numpy())

    pred_list = np.concatenate(pred_list, axis=0)
    pred_output = pred_list.reshape(-1, 2)
    output_df = pd.DataFrame(pred_output, columns=["x", "y"])
    output_df.index.name = "index"
    output_df.to_csv(f"{SAVE_DIR}/submission.csv", index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trajectory Prediction")
    parser.add_argument("--gpu", type=str, default="0", help="GPU ID to use")
    parser.add_argument("--data_version", type=str, default="v3", help="Data version")
    parser.add_argument("--n_folds", type=int, default=10, help="Number of folds")
    parser.add_argument("--fold", type=int, default=0, help="Fold number")
    parser.add_argument("--model", type=str, default="TransformerNet", help="Model to use")
    parser.add_argument("--input_dim", type=int, default=31, help="Input dimension")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--output_dim", type=int, default=60 * 2, help="Output dimension")
    parser.add_argument("--nhead", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--time_layers", type=int, default=4, help="Number of time layers")
    parser.add_argument("--agent_layers", type=int, default=4, help="Number of agent layers")
    parser.add_argument("--k_neigh", type=int, default=10, help="Number of neighbors")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale factor")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=400, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=2.5e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--step_size", type=int, default=80, help="Step size")
    parser.add_argument("--gamma", type=float, default=0.4, help="Gamma")
    parser.add_argument("--save_dir", type=str, default="./models", help="Directory to save the model")
    args = parser.parse_args()
    train(args)
