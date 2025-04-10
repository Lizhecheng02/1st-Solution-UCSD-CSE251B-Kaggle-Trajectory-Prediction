from dataset import TrajectoryDataset
from utils import load_data, train_epoch, validate, predict, evaluate_real_world_mse, create_submission
from init import get_device, seed_everything
from TrajectoryLSTM import TrajectoryLSTM
from TrajectoryLSTMAttention import TrajectoryLSTMAttention
from TrajectoryAttentionLSTM import TrajectoryAttentionLSTM
from TrajectoryTransformer1 import TrajectoryTransformer1
from TrajectoryTransformer2 import TrajectoryTransformer2
from TrajectoryTransformer3 import TrajectoryTransformer3
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import argparse
import time
import os
import json
import warnings
import numpy as np
warnings.filterwarnings("ignore")


def train(args):
    print("Training...")

    GPU = args.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU

    MODEL = args.model
    WEIGHTS_INITIALIZATION = args.weights_initialization
    FOLD = args.fold
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.num_epochs
    LR = args.lr
    MAX_NORM = args.max_norm
    SEED = args.seed
    INPUT_DIM = args.input_dim
    D_MODEL = args.d_model
    NHEAD = args.nhead
    NUM_ENCODER_LAYERS = args.num_encoder_layers
    NUM_DECODER_LAYERS = args.num_decoder_layers
    DIM_FEEDFORWARD = args.dim_feedforward
    DROPOUT = args.dropout
    MAX_LEN = args.max_len
    PRED_STEPS = args.pred_steps
    NUM_AGENT_TYPES = args.num_agent_types
    FACTOR = args.factor
    PATIENCE = args.patience
    SAVE_DIR = args.save_dir + f"/Fold{FOLD}/{MODEL}/{int(time.time())}"

    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Saving models to {SAVE_DIR}")

    seed_everything(SEED)
    device = get_device()

    train_data, test_data = load_data()

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    indices = np.arange(len(train_data))
    folds = list(kf.split(indices))

    train_idx, val_idx = folds[FOLD]
    train_split = np.array([train_data[i] for i in train_idx])
    val_split = np.array([train_data[i] for i in val_idx])

    train_dataset = TrajectoryDataset(train_split, is_train=True)
    val_dataset = TrajectoryDataset(val_split, is_train=True)

    test_dataset = TrajectoryDataset(test_data, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model_config = {
        "input_dim": INPUT_DIM,
        "d_model": D_MODEL,
        "nhead": NHEAD,
        "num_encoder_layers": NUM_ENCODER_LAYERS,
        "num_decoder_layers": NUM_DECODER_LAYERS,
        "dim_feedforward": DIM_FEEDFORWARD,
        "dropout": DROPOUT,
        "max_len": MAX_LEN,
        "pred_steps": PRED_STEPS,
        "num_agent_types": NUM_AGENT_TYPES,
        "weights_initialization": WEIGHTS_INITIALIZATION
    }

    if args.model == "TrajectoryLSTM":
        model = TrajectoryLSTM(**model_config).to(device)
    elif args.model == "TrajectoryLSTMAttention":
        model = TrajectoryLSTMAttention(**model_config).to(device)
    elif args.model == "TrajectoryAttentionLSTM":
        model = TrajectoryAttentionLSTM(**model_config).to(device)
    elif args.model == "TrajectoryTransformer1":
        model = TrajectoryTransformer1(**model_config).to(device)
    elif args.model == "TrajectoryTransformer2":
        model = TrajectoryTransformer2(**model_config).to(device)
    elif args.model == "TrajectoryTransformer3":
        model = TrajectoryTransformer3(**model_config).to(device)

    with open(f"{SAVE_DIR}/model_config.json", "w") as f:
        json.dump(model_config, f, indent=4)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=FACTOR, patience=PATIENCE)

    num_epochs = NUM_EPOCHS
    best_val_real_mse_loss = float("inf")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, max_norm=MAX_NORM, model_type=MODEL)
        val_loss = validate(model, val_loader, criterion, device)
        val_real_mse = evaluate_real_world_mse(model, val_loader, val_dataset, device)
        print(f"Train Loss: {train_loss: .6f}, Val Loss: {val_loss: .6f}, Val MSE (real): {val_real_mse: .2f} m^2")

        scheduler.step(val_loss)

        if val_real_mse < best_val_real_mse_loss:
            best_val_real_mse_loss = val_real_mse
            torch.save(model.state_dict(), f"{SAVE_DIR}/model-{epoch + 1}.pth")
            torch.save(model.state_dict(), f"{SAVE_DIR}/best-model.pth")
            print(f"The best model saved! - epoch {epoch + 1}")
        # else:
        #     torch.save(model.state_dict(), f"{SAVE_DIR}/model-{epoch + 1}.pth")
        #     print("Model saved!")

    model.load_state_dict(torch.load(f"{SAVE_DIR}/best-model.pth"))

    test_predictions = predict(model, test_dataset, device)

    create_submission(test_predictions, output_file=f"{SAVE_DIR}/submission.csv")


def inference(args):
    print("Inference...")

    SEED = args.seed
    INPUT_DIM = args.input_dim
    D_MODEL = args.d_model
    NHEAD = args.nhead
    NUM_ENCODER_LAYERS = args.num_encoder_layers
    NUM_DECODER_LAYERS = args.num_decoder_layers
    DIM_FEEDFORWARD = args.dim_feedforward
    DROPOUT = args.dropout
    MAX_LEN = args.max_len
    PRED_STEPS = args.pred_steps
    NUM_AGENT_TYPES = args.num_agent_types

    seed_everything(SEED)
    device = get_device()

    _, test_data = load_data()
    test_dataset = TrajectoryDataset(test_data, is_train=False)

    try:
        config_path = os.path.join(os.path.dirname(args.inference_checkpoint), "model_config.json")
        with open(config_path, "r") as f:
            model_config = json.load(f)
    except:
        model_config = {
            "input_dim": INPUT_DIM,
            "d_model": D_MODEL,
            "nhead": NHEAD,
            "num_encoder_layers": NUM_ENCODER_LAYERS,
            "num_decoder_layers": NUM_DECODER_LAYERS,
            "dim_feedforward": DIM_FEEDFORWARD,
            "dropout": DROPOUT,
            "max_len": MAX_LEN,
            "pred_steps": PRED_STEPS,
            "num_agent_types": NUM_AGENT_TYPES
        }

    if args.model == "TrajectoryLSTM":
        model = TrajectoryLSTM(**model_config).to(device)
    elif args.model == "TrajectoryLSTMAttention":
        model = TrajectoryLSTMAttention(**model_config).to(device)
    elif args.model == "TrajectoryAttentionLSTM":
        model = TrajectoryAttentionLSTM(**model_config).to(device)
    elif args.model == "TrajectoryTransformer1":
        model = TrajectoryTransformer1(**model_config).to(device)
    elif args.model == "TrajectoryTransformer2":
        model = TrajectoryTransformer2(**model_config).to(device)
    elif args.model == "TrajectoryTransformer3":
        model = TrajectoryTransformer3(**model_config).to(device)

    model.load_state_dict(torch.load(args.inference_checkpoint, map_location=device))
    model.eval()

    test_predictions = predict(model, test_dataset, device)

    save_dir = os.path.dirname(args.inference_checkpoint)
    create_submission(test_predictions, output_file=os.path.join(save_dir, "submission.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trajectory Prediction")
    parser.add_argument("--task", type=str, default="train", help="Task to perform: train or inference")
    parser.add_argument("--inference_checkpoint", type=str, default=None, help="Checkpoint for inference")
    parser.add_argument("--model", type=str, default="TrajectoryTransformer2", help="Model to use for training")
    parser.add_argument("--gpu", type=str, default="0", help="GPU index to use (e.g., '0')")
    parser.add_argument("--weights_initialization", type=lambda x: x.lower() in ["true", "1", "yes", "y"], default=False, help="Weight initialization method")
    parser.add_argument("--fold", type=int, default=0, help="Fold index for cross-validation (0-4)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=250, help="Number of epochs for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for training")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Max norm for gradient clipping")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--input_dim", type=int, default=5, help="Input dimension for the model")
    parser.add_argument("--d_model", type=int, default=128, help="Dimension of the model")
    parser.add_argument("--nhead", type=int, default=1, help="Number of attention heads")
    parser.add_argument("--num_encoder_layers", type=int, default=4, help="Number of encoder layers")
    parser.add_argument("--num_decoder_layers", type=int, default=4, help="Number of decoder layers")
    parser.add_argument("--dim_feedforward", type=int, default=512, help="Dimension of the feedforward network")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--max_len", type=int, default=50, help="Maximum length of the input sequence")
    parser.add_argument("--pred_steps", type=int, default=60, help="Number of prediction steps")
    parser.add_argument("--num_agent_types", type=int, default=10, help="Number of agent types")
    parser.add_argument("--factor", type=float, default=0.8, help="Factor for learning rate scheduler")
    parser.add_argument("--patience", type=int, default=50, help="Patience for learning rate scheduler")
    parser.add_argument("--save_dir", type=str, default="./models", help="Directory to save the model")
    args = parser.parse_args()
    print(args)
    if args.task == "train":
        train(args)
    elif args.task == "inference":
        inference(args)
