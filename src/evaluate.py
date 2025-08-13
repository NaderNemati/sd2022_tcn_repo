import os, argparse, yaml, json, numpy as np, pandas as pd, torch
from .metrics import to_common_columns, geodetic_to_ecef, ecef_to_enu, enu_to_ecef, ecef_to_geodetic
from .data import add_features
from .model_tcn import TCN

def sliding_windows_idx(n, seq_len, stride):
    starts = list(range(0, max(1, n - seq_len + 1), stride))
    windows = [(s, min(s+seq_len, n)) for s in starts]
    if windows and windows[-1][1] < n:
        windows.append((n - seq_len, n))
    return windows

def predict_trace(grp, model, cfg, device):
    # Featureize
    feat = add_features(grp[["phone","millisSinceGpsEpoch","latDeg","lonDeg"]])
    # build ENU reference for reconstruction
    lat = np.radians(grp["latDeg"].values); lon = np.radians(grp["lonDeg"].values)
    lat0 = float(np.median(lat)); lon0 = float(np.median(lon))
    Xb, Yb, Zb = geodetic_to_ecef(lat, lon, np.zeros_like(lat))
    Eb, Nb = [], []
    for i in range(len(grp)):
        e, n, _ = ecef_to_enu(Xb[i], Yb[i], Zb[i], lat0, lon0, 0.0)
        Eb.append(e); Nb.append(n)
    Eb = np.array(Eb); Nb = np.array(Nb)

    # windows
    seq_len = int(cfg["seq_len"]); stride = int(cfg["stride"])
    idxs = sliding_windows_idx(len(feat), seq_len, stride)

    # infer in_channels from config
    use_speed = cfg.get("use_speed", True); use_heading = cfg.get("use_heading", True); use_dt = cfg.get("use_dt", True)
    cols = ["E","N","vE","vN"]
    if use_speed: cols.append("speed")
    if use_heading: cols += ["hdg_sin","hdg_cos"]
    if use_dt: cols.append("dt")
    in_ch = len(cols)

    agg_dE = np.zeros(len(feat)); counts = np.zeros(len(feat))
    agg_dN = np.zeros(len(feat))

    for s, e in idxs:
        g = feat.iloc[s:e]
        X = np.stack([g[c].values for c in cols], axis=0).astype(np.float32)  # [C, T]
        X = torch.from_numpy(X).unsqueeze(0).to(device)
        with torch.no_grad():
            dEN = model(X).cpu().numpy()[0]  # [2, T]
        # accumulate (overlap-avg)
        agg_dE[s:e] += dEN[0]; agg_dN[s:e] += dEN[1]; counts[s:e] += 1

    dE = agg_dE / np.maximum(1, counts)
    dN = agg_dN / np.maximum(1, counts)
    # corrected ENU
    Ec = Eb + dE; Nc = Nb + dN
    # back to lat/lon
    out_lat, out_lon = [], []
    for i in range(len(Ec)):
        X2, Y2, Z2 = enu_to_ecef(Ec[i], Nc[i], 0.0, lat0, lon0, 0.0)
        lat2, lon2, _ = ecef_to_geodetic(X2, Y2, Z2)
        out_lat.append(np.degrees(lat2)); out_lon.append(np.degrees(lon2))
    res = grp[["phone","millisSinceGpsEpoch"]].copy()
    res["latDeg"] = out_lat; res["lonDeg"] = out_lon
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--config", default="configs.yaml")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    df = to_common_columns(pd.read_csv(args.input_csv)).sort_values(["phone","millisSinceGpsEpoch"])
    # Build model (infer channels from config)
    cols = ["E","N","vE","vN"]
    if cfg.get("use_speed", True): cols.append("speed")
    if cfg.get("use_heading", True): cols += ["hdg_sin","hdg_cos"]
    if cfg.get("use_dt", True): cols.append("dt")
    in_ch = len(cols)

    model = TCN(in_channels=in_ch, channels=cfg["tcn_channels"],
                kernel_size=cfg["tcn_kernel_size"], dropout=cfg["tcn_dropout"], out_channels=2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.to(device); model.eval()

    outs = []
    for ph, grp in df.groupby("phone"):
        outs.append(predict_trace(grp, model, cfg, device))
    pred = pd.concat(outs, ignore_index=True)
    pred.to_csv(args.out_csv, ind

