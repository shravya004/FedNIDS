"""
scripts/sanity_check_train.py

Quick CENTRALIZED smoke test -- this is NOT the federated training loop
(that's Member 3's job). It only proves the chain works end-to-end:

    data/processed/clients/client_<id>/{train,val}.csv
        -> FLNetDataset
        -> HybridGRUTransformer
        -> trains, and loss/accuracy improve

Run AFTER scripts/run_preprocessing.py:
    python scripts/sanity_check_train.py --client_id 1 --epochs 5
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

import config
from src.datasets.flnet_dataset import load_client_loaders
from src.models.hybrid_model import HybridGRUTransformer
from src.utils.seed import set_seed


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model.predict(x)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(y.cpu().numpy().tolist())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return acc, f1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, seq_len, input_dim = load_client_loaders(
        args.client_id, batch_size=args.batch_size
    )
    num_classes = max(
        int(train_loader.dataset.y.max()) + 1,
        int(val_loader.dataset.y.max()) + 1,
    )
    print(f"Client {args.client_id}: seq_len={seq_len}, input_dim={input_dim}, num_classes={num_classes}")

    model = HybridGRUTransformer(input_dim=input_dim, seq_len=seq_len, num_classes=num_classes).to(device)
    print("Model config:", model.get_config())

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        val_acc, val_f1 = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}/{args.epochs} | train_loss={avg_loss:.4f} | "
              f"val_acc={val_acc:.4f} | val_f1={val_f1:.4f}")

    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = config.ARTIFACTS_DIR / f"hybrid_model_client{args.client_id}_sanity.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"\nSaved sanity-check checkpoint -> {ckpt_path}")


if __name__ == "__main__":
    main()
