import numpy as np
import torch
import torch.nn as nn
import os
import argparse
import warnings
from torch_geometric.data import DataLoader, Batch
from dataset import TrajectoryDatasetTrain
from init import seed_everything, get_device
from models import LSTMNet, TransformerNet
from sklearn.model_selection import KFold
warnings.filterwarnings("ignore")


def auto_infer_model(model_path, input_dim, output_dim, model_name):
    print("Auto-inferring model parameters...")
    hidden_dims = [64, 128, 256]
    nheads = [2, 4, 8]
    nlayers = [2, 4, 6, 8]

    for hidden in hidden_dims:
        for nhead in nheads:
            for nlayer in nlayers:
                try:
                    if model_name == "TransformerNet":
                        model = TransformerNet(
                            input_dim=input_dim,
                            model_dim=hidden,
                            output_dim=output_dim,
                            nhead=nhead,
                            num_layers=nlayer
                        )
                    elif model_name == "LSTMNet":
                        model = LSTMNet(
                            input_dim=input_dim,
                            hidden_dim=hidden,
                            output_dim=output_dim
                        )
                    else:
                        raise ValueError("Unsupported model name for auto inference")
                    model.load_state_dict(torch.load(model_path))
                    print(f"Inferred hidden_dim={hidden}, nhead={nhead}, num_layers={nlayer}")
                    return model.to(get_device())
                except Exception:
                    continue

    raise RuntimeError("Failed to infer model parameters. Please provide them manually.")


def validate(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    seed_everything()
    device = get_device()

    model_path = os.path.join(args.save_dir, "best_model.pt")
    print(f"Loading model from: {model_path}")
    assert os.path.exists(model_path), "best_model.pt not found!"

    train_npz = np.load("../data/train.npz")
    train_data = train_npz["data"]

    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=777)
    folds = list(kf.split(train_data))
    _, val_idx = folds[args.fold]
    val_dataset = TrajectoryDatasetTrain([train_data[i] for i in val_idx], scale=args.scale, augment=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: Batch.from_data_list(x))

    model = auto_infer_model(model_path, args.input_dim, args.output_dim, args.model)
    model.eval()

    mae_loss = 0.0
    mse_loss = 0.0
    count = 0

    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            pred = model(batch)
            y = batch.y.view(batch.num_graphs, 60, 2)

            pred = pred * batch.scale.view(-1, 1, 1) + batch.origin.unsqueeze(1)
            y = y * batch.scale.view(-1, 1, 1) + batch.origin.unsqueeze(1)

            mae_loss += nn.L1Loss(reduction="sum")(pred, y).item()
            mse_loss += nn.MSELoss(reduction="sum")(pred, y).item()
            count += batch.num_graphs * 60

    mae_loss /= count
    mse_loss /= count

    print(f"\n[Validation Score on Best Model]")
    print(f"MAE: {mae_loss:.6f}")
    print(f"MSE: {mse_loss:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Best Model on Validation Set")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--n_folds", type=int, default=10)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--model", type=str, default="TransformerNet")
    parser.add_argument("--input_dim", type=int, default=6)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--output_dim", type=int, default=120)
    parser.add_argument("--nhead", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--save_dir", type=str, default="./models")
    args = parser.parse_args()
    validate(args)
