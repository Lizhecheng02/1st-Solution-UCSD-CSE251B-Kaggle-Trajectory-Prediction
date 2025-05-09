import os
import torch
import matplotlib.pyplot as plt


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


def plot_trajectory(ax, pred, gt, title=None):
    ax.cla()
    ax.plot(pred[0, :60, 0], pred[0, :60, 1], color="red", label="Predicted Future Trajectory")

    ax.plot(gt[0, :60, 0], gt[0, :60, 1], color="blue", label="Ground Truth Future Trajectory")

    x_max = max(pred[..., 0].max(), gt[..., 0].max())
    x_min = min(pred[..., 0].min(), gt[..., 0].min())
    y_max = max(pred[..., 1].max(), gt[..., 1].max())
    y_min = min(pred[..., 1].min(), gt[..., 1].min())

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")

    if title:
        ax.set_title(title)

    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)


def plot_loss_curve(train_losses, val_losses, val_maes, val_mses, fold, total_folds, model_type):
    os.makedirs("figures", exist_ok=True)
    mark_interval = max(1, len(train_losses) // 10)

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, marker="^", markevery=mark_interval, color="tab:blue", label="Train Loss")
    plt.plot(val_losses, marker="s", markevery=mark_interval, color="tab:orange", label="Validation Loss")
    plt.plot(val_maes, marker="o", markevery=mark_interval, color="tab:green", label="Validation MAE")
    plt.plot(val_mses, marker="x", markevery=mark_interval, color="tab:red", label="Validation MSE")

    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.title(f"{model_type} Losses and Metrics (Fold {fold}/{total_folds})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"figures/{model_type}-Fold-{fold}-Metrics.pdf", format="pdf", dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close()
