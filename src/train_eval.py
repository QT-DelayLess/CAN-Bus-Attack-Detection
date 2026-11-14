"""
Training and Evaluation pipeline for the 1D-CNN CAN Bus Intrusion Detection System.

Performs:
  - Preprocessing of the HCRL Car-Hacking Dataset (DoS, Fuzzy, RPM, Gear)
  - Stratified 80/20 Train/Test split
  - 10-epoch training of the 1D-CNN model with automatic CUDA/CPU selection
  - Multi-class evaluation (Accuracy, Precision, Recall, F1-Score)
  - Saving summary table to `results/metrics.txt`
  - Generating high-resolution plots:
      1. `results/confusion_matrix.png`
      2. `results/training_curves.png`
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Allow running both `python src/train_eval.py` and `python -m src.train_eval`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import CAN1DCNN, get_device
from src.preprocess import CLASS_NAMES, load_and_preprocess_data


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Evaluate model on a DataLoader and return (loss, accuracy, y_true, y_pred)."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            batch_size = batch_y.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    avg_loss = total_loss / max(1, total_samples)
    acc = accuracy_score(y_true, y_pred)
    return avg_loss, acc, y_true, y_pred


def plot_training_curves(
    history: Dict[str, List[float]],
    output_path: str = "results/training_curves.png",
) -> None:
    """Generate high-resolution training & validation Loss and Accuracy curves."""
    epochs = range(1, len(history["train_loss"]) + 1)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # Loss subplot
    axes[0].plot(epochs, history["train_loss"], marker="o", linewidth=2, label="Train Loss", color="#1f77b4")
    axes[0].plot(epochs, history["val_loss"], marker="s", linewidth=2, linestyle="--", label="Validation Loss", color="#d62728")
    axes[0].set_title("1D-CNN Training & Validation Loss", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Cross-Entropy Loss", fontsize=11)
    axes[0].set_xticks(list(epochs))
    axes[0].legend(frameon=True, fontsize=10)
    axes[0].grid(True, linestyle=":", alpha=0.7)

    # Accuracy subplot
    train_acc_pct = [a * 100.0 for a in history["train_acc"]]
    val_acc_pct = [a * 100.0 for a in history["val_acc"]]
    axes[1].plot(epochs, train_acc_pct, marker="o", linewidth=2, label="Train Accuracy", color="#2ca02c")
    axes[1].plot(epochs, val_acc_pct, marker="s", linewidth=2, linestyle="--", label="Validation Accuracy", color="#ff7f0e")
    axes[1].set_title("1D-CNN Training & Validation Accuracy", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Accuracy (%)", fontsize=11)
    axes[1].set_xticks(list(epochs))
    axes[1].legend(frameon=True, fontsize=10)
    axes[1].grid(True, linestyle=":", alpha=0.7)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Results] Saved training curves plot to {output_path}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str = "results/confusion_matrix.png",
) -> None:
    """Generate a high-resolution multi-class confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]:,}\n({cm_norm[i, j] * 100:.1f}%)"

    fig, ax = plt.subplots(figsize=(8.5, 7), dpi=300)
    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar_kws={"label": "Number of CAN Frames"},
        ax=ax,
        annot_kws={"size": 10, "weight": "bold"},
    )
    ax.set_title("Multi-Class Confusion Matrix (1D-CNN CAN Bus IDS)", fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Predicted Class", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Class", fontsize=11, fontweight="bold")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Results] Saved confusion matrix plot to {output_path}")


def save_metrics_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    history: Dict[str, List[float]],
    device: torch.device,
    sample_size: Optional[int],
    elapsed_sec: float,
    output_path: str = "results/metrics.txt",
) -> None:
    """Save a formatted summary table of metrics to `results/metrics.txt`."""
    acc = accuracy_score(y_true, y_pred)
    prec_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    cls_report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred)

    lines = [
        "=" * 78,
        "ANOMALY AND ATTACK DETECTION IN INDUSTRIAL AND IN-VEHICLE NETWORKS",
        "1D-CNN Multi-Class Evaluation Summary (HCRL Car-Hacking Dataset)",
        "=" * 78,
        f"Device Used          : {device}",
        f"Sample Size per File : {sample_size if sample_size is not None else 'Full Dataset'}",
        f"Test Set Size        : {len(y_true):,} frames (20% stratified split)",
        f"Epochs Trained       : {len(history['train_loss'])}",
        f"Total Training Time  : {elapsed_sec:.2f} seconds",
        "-" * 78,
        "OVERALL PERFORMANCE METRICS",
        "-" * 78,
        f"{'Metric':<28} | {'Weighted Avg':<18} | {'Macro Avg':<18}",
        "-" * 78,
        f"{'Accuracy':<28} | {acc * 100:>16.2f}% | {acc * 100:>16.2f}%",
        f"{'Precision':<28} | {prec_weighted * 100:>16.2f}% | {prec_macro * 100:>16.2f}%",
        f"{'Recall':<28} | {rec_weighted * 100:>16.2f}% | {rec_macro * 100:>16.2f}%",
        f"{'F1-Score':<28} | {f1_weighted * 100:>16.2f}% | {f1_macro * 100:>16.2f}%",
        "-" * 78,
        "PER-CLASS CLASSIFICATION REPORT",
        "-" * 78,
        cls_report.strip(),
        "-" * 78,
        "CONFUSION MATRIX (Rows: True Class, Columns: Predicted Class)",
        "-" * 78,
        f"{'Class':<14}" + "".join(f"{name:>14}" for name in CLASS_NAMES),
    ]
    for idx, name in enumerate(CLASS_NAMES):
        row_str = f"{name:<14}" + "".join(f"{cm[idx, j]:>14,}" for j in range(len(CLASS_NAMES)))
        lines.append(row_str)

    lines.extend([
        "-" * 78,
        "EPOCH-BY-EPOCH TRAINING HISTORY",
        "-" * 78,
        f"{'Epoch':<8} | {'Train Loss':<14} | {'Train Acc (%)':<15} | {'Val Loss':<14} | {'Val Acc (%)':<15}",
        "-" * 78,
    ])
    for ep in range(len(history["train_loss"])):
        lines.append(
            f"{ep + 1:<8} | "
            f"{history['train_loss'][ep]:<14.5f} | "
            f"{history['train_acc'][ep] * 100:<15.2f} | "
            f"{history['val_loss'][ep]:<14.5f} | "
            f"{history['val_acc'][ep] * 100:<15.2f}"
        )
    lines.append("=" * 78)

    report_text = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print("\n" + report_text)
    print(f"[Results] Saved evaluation summary to {output_path}")


def train_and_evaluate(
    data_dir: str = "data",
    results_dir: str = "results",
    sample_size: Optional[int] = 50000,
    epochs: int = 10,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
) -> None:
    set_seed(42)
    device = get_device()
    print(f"[Setup] Using compute device: {device}")

    # 1. Load and preprocess dataset
    X, y, _ = load_and_preprocess_data(data_dir=data_dir, sample_size=sample_size)

    # 2. Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )
    print(f"[Split] Train set: {len(X_train):,} samples | Test set: {len(X_test):,} samples")

    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 3. Initialize 1D-CNN model, loss criterion, and optimizer
    model = CAN1DCNN(input_features=X.shape[1], num_classes=len(CLASS_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    # 4. Train for 10 epochs
    print(f"[Train] Starting 1D-CNN training for {epochs} epochs...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            batch_sz = batch_y.size(0)
            running_loss += loss.item() * batch_sz
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_sz

        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)

        val_loss, val_acc, _, _ = evaluate_loader(model, test_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"  Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%"
        )

    elapsed = time.time() - start_time

    # 5. Final evaluation & artifacts generation
    _, _, y_true, y_pred = evaluate_loader(model, test_loader, criterion, device)

    os.makedirs(results_dir, exist_ok=True)
    save_metrics_report(
        y_true=y_true,
        y_pred=y_pred,
        history=history,
        device=device,
        sample_size=sample_size,
        elapsed_sec=elapsed,
        output_path=os.path.join(results_dir, "metrics.txt"),
    )
    plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        output_path=os.path.join(results_dir, "confusion_matrix.png"),
    )
    plot_training_curves(
        history=history,
        output_path=os.path.join(results_dir, "training_curves.png"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate 1D-CNN for CAN Bus Attack Detection."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing HCRL Car-Hacking CSV datasets.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory to save metrics and plots.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=50000,
        help="Rows per CSV file to load (set to 0 or negative for full dataset).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Mini-batch size (default: 256).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 0.001).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    effective_sample_size = None if (args.sample_size is not None and args.sample_size <= 0) else args.sample_size
    train_and_evaluate(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        sample_size=effective_sample_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
