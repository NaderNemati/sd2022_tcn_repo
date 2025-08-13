import os, argparse, yaml, json, math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from .data import prepare_training_tables, WindowedDataset
from .model_tcn import TCN

def huber_loss(pred, target, delta=1.0):
    # pred/target: [B, 2, T]
    err = pred - target
    abs_err = torch.abs(err)
    quad = torch.minimum(abs_err, torch.tensor(delta, device=pred.device))
    lin = abs_err - quad
    return ((0.5 * quad**2) + delta * lin).mean()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--gt_csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--bs", type=int, default=None)
    ap.add_argument("--config", default="configs.yaml")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    seq_len = int(cfg["seq_len"]); stride = int(cfg["stride"])
    batch_size = int(args.bs or cfg["batch_size"])
    epochs = int(args.epochs or cfg["epochs"])

    # prepare data table with features + targets
    table = prepare_training_tables(args.train_csv, args.gt_csv)

    feat_flags = {"use_speed": cfg.get("use_speed", True),
                  "use_heading": cfg.get("use_heading", True),
                  "use_dt": cfg.get("use_dt", True)}
    ds = WindowedDataset(table, seq_len=seq_len, stride=stride, feature_flags=feat_flags)

    # infer input channels
    dummy_X, dummy_y = ds[0]
    in_ch = dummy_X.shape[0]
    model = TCN(in_channels=in_ch, channels=cfg["tcn_channels"],
                kernel_size=cfg["tcn_kernel_size"], dropout=cfg["tcn_dropout"], out_channels=2)

    # split
    vsz = int(len(ds) * float(cfg["val_split"]))
    trsz = len(ds) - vsz
    tr_ds, va_ds = random_split(ds, [trsz, vsz], generator=torch.Generator().manual_seed(42))

    tr_ld = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    va_ld = DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))

    best = math.inf; patience = 8; bad = 0
    for epoch in range(1, epochs+1):
        model.train()
        tr_loss = 0.0
        for X, y in tr_ld:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            pred = model(X)
            loss = huber_loss(pred, y, delta=float(cfg["huber_delta"]))
            loss.backward()
            opt.step()
            tr_loss += loss.item() * X.size(0)
        tr_loss /= len(tr_ld.dataset)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for X, y in va_ld:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                loss = huber_loss(pred, y, delta=float(cfg["huber_delta"]))
                va_loss += loss.item() * X.size(0)
        va_loss /= len(va_ld.dataset)

        print(f"epoch {epoch:02d} | train_huber={tr_loss:.4f} | val_huber={va_loss:.4f}")

        if va_loss < best:
            best, bad = va_loss, 0
            torch.save(model.state_dict(), os.path.join(args.out, "best.pt"))
        else:
            bad += 1
        if bad >= patience:
            print("Early stopping."); break

    with open(os.path.join(args.out, "training_summary.json"), "w") as f:
        json.dump({"best_val_huber": best}, f, indent=2)

if __name__ == "__main__":
    main()

