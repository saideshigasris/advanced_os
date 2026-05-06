"""
Training script for the GNN deadlock predictor.
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.datasets import SyntheticGraphDataset
from src.datasets.graph_loader import GraphDataLoader
from src.gnn import DeadlockGNN


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    preds, truths = [], []
    for data, labels in tqdm(loader, desc="Train", leave=False):
        data = data.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds.extend((out > 0.5).int().cpu().numpy().flatten())
        truths.extend(labels.int().cpu().numpy().flatten())
    acc = accuracy_score(truths, preds)
    return total_loss / len(loader), acc


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    preds, truths = [], []
    with torch.no_grad():
        for data, labels in loader:
            data = data.to(device)
            labels = labels.to(device)
            out = model(data)
            loss = criterion(out, labels)
            total_loss += loss.item()
            preds.extend((out > 0.5).int().cpu().numpy().flatten())
            truths.extend(labels.int().cpu().numpy().flatten())
    acc = accuracy_score(truths, preds)
    p, r, f1, _ = precision_recall_fscore_support(truths, preds, average="binary")
    return total_loss / len(loader), acc, p, r, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--max-nodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=str, default="checkpoints/gnn_model.pt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Generating synthetic dataset...")
    dataset = SyntheticGraphDataset(
        num_samples=args.samples,
        min_processes=10,
        max_processes=args.max_nodes,
        balanced=True,
        seed=args.seed,
    )
    train_loader, val_loader = GraphDataLoader.from_synthetic(
        dataset, batch_size=args.batch_size, train_ratio=0.8
    )

    model = DeadlockGNN(
        in_channels=3,
        hidden_channels=args.hidden,
        num_layers=args.layers,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    best_acc = 0

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_p, val_r, val_f1 = eval_epoch(model, val_loader, criterion, device)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), args.save)

        print(
            f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} P: {val_p:.4f} R: {val_r:.4f} F1: {val_f1:.4f}"
        )

    print(f"\nBest validation accuracy: {best_acc:.4f}")
    print(f"Model saved to {args.save}")


if __name__ == "__main__":
    main()
