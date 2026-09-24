"""
SOL Swing - Backtest Interactivo
===================================
Te pregunta:
- Rango de RSI de entrada (low, high)
- Cuantos dias hacia atras testear

Corre con los demas parametros ya validados: VP_window=90d, stop=6%,
target=VAH, RSI salida sobrecompra=75. Al final te da winrate,
expectancy, equity y guarda los trades en CSV.

Requisitos:
    pip install requests pandas numpy

Uso:
    python3 sol_swing_backtest_interactivo.py
"""

import time
import requests
import pandas as pd
import numpy as np

SYMBOL = "SOLUSDT"
INTERVAL = "1d"
BASE_URL = "https://api.binance.com/api/v3/klines"

VP_WINDOW = 90
RSI_LEN = 14
RSI_EXIT_OVERBOUGHT = 75
STOP_PCT = 0.06
HVN_PROXIMITY_PCT = 0.015
EMA_FAST = 50
EMA_SLOW = 200
VP_BINS = 24


def ask_int(prompt, default):
    raw = input(f"{prompt} (Enter = {default}): ").strip()
    if raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        print("Numero invalido, usando default.")
        return default


def ask_float(prompt, default):
    raw = input(f"{prompt} (Enter = {default}): ").strip()
    if raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        print("Numero invalido, usando default.")
        return default


# -------------------- DATOS / INDICADORES --------------------
def fetch_klines(symbol, interval, days):
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
        time.sleep(0.25)
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


def rolling_volume_profile(df, window=VP_WINDOW, bins=VP_BINS):
    poc_l, vah_l, val_l, hvn_l = [], [], [], []
    for i in range(len(df)):
        if i < window:
            poc_l.append(np.nan); vah_l.append(np.nan); val_l.append(np.nan); hvn_l.append([])
            continue
        w = df.iloc[i-window:i+1]
        lo, hi = w["low"].min(), w["high"].max()
        if hi == lo:
            poc_l.append(np.nan); vah_l.append(np.nan); val_l.append(np.nan); hvn_l.append([])
            continue
        edges = np.linspace(lo, hi, bins+1)
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
            if cum >= 0.70*total: break
        vah = centers[max(vabins)]; val = centers[min(vabins)]
        avg = total/bins
        hvns = centers[vol_bin > 1.3*avg].tolist()
        poc_l.append(poc); vah_l.append(vah); val_l.append(val); hvn_l.append(hvns)
    df["poc"]=poc_l; df["vah"]=vah_l; df["val"]=val_l; df["hvns"]=hvn_l
    return df


def near_any(price, levels, pct=HVN_PROXIMITY_PCT):
    if not levels: return False
    return any(abs(price-l)/price <= pct for l in levels)


# -------------------- BACKTEST --------------------
def run_backtest(df, rsi_lo, rsi_hi, stop_pct=STOP_PCT):
    trades = []
    in_pos = False
    ep = ed = epoc = None
    for i in range(EMA_SLOW + VP_WINDOW, len(df)):
        row = df.iloc[i]
        price = row["close"]
        bias_ok = (price > row["ema_slow"]) and (row["ema_fast"] > row["ema_slow"])
        rsi_reset = rsi_lo <= row["rsi"] <= rsi_hi
        near_hvn = near_any(price, row["hvns"] + [row["poc"]])

        if not in_pos:
            if bias_ok and rsi_reset and near_hvn:
                in_pos = True; ep = price; ed = row["open_time"]; epoc = row["poc"]
        else:
            stop_hit = price < epoc * (1 - stop_pct)
            target_hit = row["vah"] and price >= row["vah"]
            ob = row["rsi"] >= RSI_EXIT_OVERBOUGHT
            trend_broken = row["ema_fast"] < row["ema_slow"]
            if stop_hit or target_hit or ob or trend_broken:
                pnl_pct = (price - ep) / ep * 100
                reason = ("stop_invalidacion" if stop_hit else
                          "target_vah" if target_hit else
                          "rsi_sobrecompra" if ob else "cambio_tendencia")
                trades.append({"entry_date": ed, "exit_date": row["open_time"],
                                "entry_price": ep, "exit_price": price,
                                "pnl_pct": pnl_pct, "days_held": (row["open_time"]-ed).days,
                                "exit_reason": reason})
                in_pos = False
    return pd.DataFrame(trades)


def summarize(trades, rsi_lo, rsi_hi, lookback_days):
    print(f"\n{'='*55}")
    print(f"RESULTADOS — RSI {rsi_lo}-{rsi_hi} | {lookback_days} dias de historia")
    print(f"{'='*55}")
    if trades.empty:
        print("No se generaron trades con estos parametros.")
        return
    wins = trades[trades.pnl_pct > 0]
    losses = trades[trades.pnl_pct <= 0]
    winrate = len(wins)/len(trades)*100
    avg_win = wins.pnl_pct.mean() if not wins.empty else 0
    avg_loss = losses.pnl_pct.mean() if not losses.empty else 0
    expectancy = trades.pnl_pct.mean()
    equity = 100
    for p in trades.pnl_pct:
        equity *= (1+p/100)

    print(f"Trades:            {len(trades)}")
    print(f"Winrate:           {winrate:.1f}%")
    print(f"Avg win/loss:      {avg_win:.2f}% / {avg_loss:.2f}%")
    print(f"Expectancy:        {expectancy:.2f}%")
    print(f"Equity (100->):    {equity:.2f}")
    print(f"Dias prom. en pos: {trades['days_held'].mean():.1f}")
    print(f"\nRazones de salida:")
    print(trades["exit_reason"].value_counts())


if __name__ == "__main__":
    print("=== Configuración del backtest ===")
    rsi_lo = ask_int("RSI mínimo de entrada", 45)
    rsi_hi = ask_int("RSI máximo de entrada", 60)
    lookback_days = ask_int("Días de historia a testear", 730)

    if rsi_lo >= rsi_hi:
        print("RSI minimo debe ser menor al maximo. Usando 45-60 por default.")
        rsi_lo, rsi_hi = 45, 60

    print(f"\nDescargando {lookback_days} días de {SYMBOL}...")
    df = fetch_klines(SYMBOL, INTERVAL, lookback_days)
    print(f"{len(df)} velas descargadas ({df['open_time'].min().date()} a {df['open_time'].max().date()})")

    if len(df) < EMA_SLOW + VP_WINDOW + 30:
        print(f"\n⚠ Advertencia: con solo {len(df)} velas, apenas alcanza para EMA200+VP90.")
        print("  Los resultados serán poco confiables. Prueba con más días.")

    df = add_ema(df, EMA_FAST, "ema_fast")
    df = add_ema(df, EMA_SLOW, "ema_slow")
    df = add_rsi(df)
    df = rolling_volume_profile(df)

    trades = run_backtest(df, rsi_lo, rsi_hi)
    summarize(trades, rsi_lo, rsi_hi, lookback_days)

    if not trades.empty:
        fname = f"Data/interactive/sol_backtest_rsi{rsi_lo}-{rsi_hi}_{lookback_days}d.csv"
        trades.to_csv(fname, index=False)
        print(f"\nGuardado: {fname}")
