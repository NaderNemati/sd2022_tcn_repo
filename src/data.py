import numpy as np, pandas as pd

# WGS84
_A = 6378137.0
_F = 1/298.257223563
_E2 = _F*(2 - _F)

def to_common_columns(df):
    cols = df.columns
    rename = {}
    if "UnixTimeMillis" in cols and "millisSinceGpsEpoch" not in cols:
        rename["UnixTimeMillis"] = "millisSinceGpsEpoch"
    if "LatitudeDegrees" in cols and "latDeg" not in cols:
        rename["LatitudeDegrees"] = "latDeg"
    if "LongitudeDegrees" in cols and "lonDeg" not in cols:
        rename["LongitudeDegrees"] = "lonDeg"
    return df.rename(columns=rename)

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371008.8
    lat1 = np.radians(lat1); lon1 = np.radians(lon1)
    lat2 = np.radians(lat2); lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

def compute_gsdc_score(pred_df, gt_df):
    p = to_common_columns(pred_df).copy()
    g = to_common_columns(gt_df).copy()
    key = ["phone","millisSinceGpsEpoch"]
    m = p.merge(g[key + ["latDeg","lonDeg"]].rename(columns={"latDeg":"gt_lat","lonDeg":"gt_lon"}),
                on=key, how="inner")
    if len(m)==0:
        return {"score": None, "detail": {}}
    m["err_m"] = haversine_m(m["latDeg"], m["lonDeg"], m["gt_lat"], m["gt_lon"])
    per_phone = {}
    for ph, grp in m.groupby("phone"):
        e = np.sort(grp["err_m"].values)
        p50 = np.percentile(e, 50); p95 = np.percentile(e, 95)
        per_phone[ph] = {"p50": float(p50), "p95": float(p95), "mean": float((p50+p95)/2)}
    score = float(np.mean([v["mean"] for v in per_phone.values()]))
    return {"score": score, "detail": per_phone}

def geodetic_to_ecef(lat, lon, h):
    sin_lat = np.sin(lat); cos_lat = np.cos(lat)
    N = _A / np.sqrt(1 - _E2 * sin_lat**2)
    X = (N + h) * cos_lat * np.cos(lon)
    Y = (N + h) * cos_lat * np.sin(lon)
    Z = (N * (1 - _E2) + h) * sin_lat
    return X, Y, Z

def ecef_to_enu(x, y, z, lat0, lon0, h0):
    x0, y0, z0 = geodetic_to_ecef(lat0, lon0, h0)
    dx, dy, dz = x - x0, y - y0, z - z0
    slat, clat = np.sin(lat0), np.cos(lat0)
    slon, clon = np.sin(lon0), np.cos(lon0)
    t = np.array([
        [-slon, clon, 0],
        [-clon*slat, -slon*slat, clat],
        [clon*clat, slon*clat, slat]
    ])
    de, dn, du = t @ np.array([dx, dy, dz])
    return de, dn, du

def enu_to_ecef(e, n, u, lat0, lon0, h0):
    x0, y0, z0 = geodetic_to_ecef(lat0, lon0, h0)
    slat, clat = np.sin(lat0), np.cos(lat0)
    slon, clon = np.sin(lon0), np.cos(lon0)
    R = np.array([
        [-slon, -clon*slat,  clon*clat],
        [ clon, -slon*slat,  slon*clat],
        [ 0.0,        clat,       slat]
    ])
    dx, dy, dz = R @ np.array([e, n, u])
    return x0 + dx, y0 + dy, z0 + dz

def ecef_to_geodetic(x, y, z):
    lon = np.arctan2(y, x)
    p = np.sqrt(x*x + y*y)
    lat = np.arctan2(z, p*(1 - _E2))
    for _ in range(5):
        N = _A / np.sqrt(1 - _E2*np.sin(lat)**2)
        h = p/np.cos(lat) - N
        lat = np.arctan2(z, p*(1 - _E2*N/(N+h)))
    N = _A / np.sqrt(1 - _E2*np.sin(lat)**2)
    h = p/np.cos(lat) - N
    return lat, lon, h

