"""
SOL Swing - Señal en Vivo
==========================
Corre esto cada vez que quieras chequear si hay setup activo HOY.
Usa los parámetros validados out-of-sample: VP_window=90d, RSI entry 45-60, stop=6%.

Requisitos:
    pip install requests pandas numpy

Uso:
    python3 sol_live_signal.py
"""

import time
import requests
import pandas as pd
import numpy as np

SYMBOL = "SOLUSDT"
INTERVAL = "1d"
BASE_URL = "https://api.binance.com/api/v3/klines"

# --- Parámetros validados (grid search train/test, ver análisis previo) ---
VP_WINDOW = 90
RSI_LEN = 14
RSI_ENTRY_LOW = 45
RSI_ENTRY_HIGH = 60
RSI_EXIT_OVERBOUGHT = 75
STOP_PCT = 0.06
HVN_PROXIMITY_PCT = 0.015
EMA_FAST = 50
EMA_SLOW = 200
VP_BINS = 24
LOOKBACK_DAYS = 400  # suficiente para EMA200 + VP_WINDOW con margen


def fetch_klines(symbol=SYMBOL, interval=INTERVAL, days=LOOKBACK_DAYS):
    end_time = int(time.time() * 1000)
    start_time = end_time - days * 24 * 60 * 60 * 1000
    all_rows = []
    while start_time < end_time:
        params = {"symbol": symbol, "interval": interval, "startTime": start_time, "limit": 1000}
        resp = requests.get(BASE_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        all_rows.extend(data)
        start_time = data[-1][6] + 1
        if len(data) < 1000:
            break
        time.sleep(0.3)
    cols = ["open_time","open","high","low","close","volume","close_time",
            "quote_asset_volume","num_trades","taker_buy_base","taker_buy_quote","ignore"]
    df = pd.DataFrame(all_rows, columns=cols)
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    return df[["open_time","open","high","low","close","volume"]].drop_duplicates(subset="open_time").reset_index(drop=True)


def add_ema(df, length, col):
    df[col] = df["close"].ewm(span=length, adjust=False).mean()
    return df


def add_rsi(df, length=RSI_LEN):
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = (100 - 100/(1+rs)).fillna(50)
    return df


def volume_profile_latest(df, window=VP_WINDOW, bins=VP_BINS):
    """Calcula POC/VAH/VAL/HVN solo para la última ventana (no rolling completo, más rápido)."""
    w = df.iloc[-window:]
    lo, hi = w["low"].min(), w["high"].max()
    edges = np.linspace(lo, hi, bins + 1)
    vol_bin = np.zeros(bins)
    for _, r in w.iterrows():
        rlo, rhi, rvol = r["low"], r["high"], r["volume"]
        if rhi == rlo:
            idx = min(max(np.searchsorted(edges, rlo)-1, 0), bins-1)
            vol_bin[idx] += rvol
            continue
        ov_lo = np.maximum(edges[:-1], rlo)
        ov_hi = np.minimum(edges[1:], rhi)
        ov = np.clip(ov_hi - ov_lo, 0, None)
        vol_bin += (ov/(rhi-rlo)) * rvol
    centers = (edges[:-1] + edges[1:]) / 2
    total = vol_bin.sum()
    poc = centers[np.argmax(vol_bin)]
    order = np.argsort(vol_bin)[::-1]
    cum = 0; vabins = []
    for idx in order:
        cum += vol_bin[idx]; vabins.append(idx)
        if cum >= 0.70 * total:
            break
    vah = centers[max(vabins)]
    val = centers[min(vabins)]
    avg = total / bins
    hvns = centers[vol_bin > 1.3*avg].tolist()
    return poc, vah, val, hvns


def near_any(price, levels, pct=HVN_PROXIMITY_PCT):
    if not levels:
        return False
    return any(abs(price-l)/price <= pct for l in levels)


if __name__ == "__main__":
    print(f"Descargando datos recientes de {SYMBOL}...")
    df = fetch_klines()
    df = add_ema(df, EMA_FAST, "ema_fast")
    df = add_ema(df, EMA_SLOW, "ema_slow")
    df = add_rsi(df)

    poc, vah, val, hvns = volume_profile_latest(df)
    last = df.iloc[-1]
    price = last["close"]

    bias_ok = (price > last["ema_slow"]) and (last["ema_fast"] > last["ema_slow"])
    rsi_reset = RSI_ENTRY_LOW <= last["rsi"] <= RSI_ENTRY_HIGH
    near_hvn = near_any(price, hvns + [poc])
    setup_active = bias_ok and rsi_reset and near_hvn

    print(f"\n{'='*55}")
    print(f"SOL SWING - SEÑAL AL {last['open_time'].date()}")
    print(f"{'='*55}")
    print(f"Precio actual:        {price:.2f}")
    print(f"EMA50 / EMA200:       {last['ema_fast']:.2f} / {last['ema_slow']:.2f}")
    print(f"RSI(14):              {last['rsi']:.2f}")
    print(f"POC / VAH / VAL:      {poc:.2f} / {vah:.2f} / {val:.2f}")
    print(f"HVNs cercanos:        {[round(h,2) for h in hvns]}")
    print(f"{'-'*55}")
    print(f"Bias alcista (EMA50>EMA200 y precio>EMA200): {'SI' if bias_ok else 'NO'}")
    print(f"RSI en zona de entrada (45-60):              {'SI' if rsi_reset else 'NO'}")
    print(f"Precio cerca de HVN/POC:                     {'SI' if near_hvn else 'NO'}")
    print(f"{'='*55}")
    if setup_active:
        stop_level = poc * (1 - STOP_PCT)
        print(f">>> SETUP ACTIVO — condiciones de compra cumplidas <<<")
        print(f"Referencia de entrada: {price:.2f}")
        print(f"Stop sugerido (-6% del POC): {stop_level:.2f}")
        print(f"Target (VAH del rango):      {vah:.2f}")
    else:
        faltantes = []
        if not bias_ok: faltantes.append("bias alcista")
        if not rsi_reset: faltantes.append("RSI en zona 45-60")
        if not near_hvn: faltantes.append("proximidad a HVN/POC")
        print(f">>> SIN SETUP — falta: {', '.join(faltantes)} <<<")
