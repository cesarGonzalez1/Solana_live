"""
SOL Intradia v3 - VWAP Reclaim: historico completo + train/test + diagnostico
================================================================================
Cambios vs v2:
1. Descarga TODO el historico disponible por timeframe (desde ~2020-08-11),
   no solo 90-365 dias. Mas oportunidades de reclaim = muestra mas grande.
2. Split train/test explicito (igual que el swing): antes de 2024-06-01 =
   train, despues = test out-of-sample. Reporta ambos por separado.
3. Corre cada timeframe CON y SIN el filtro de volumen de confirmacion,
   para diagnosticar si el filtro de volumen es el que esta matando la
   muestra o si el setup en si es raro.

Esto tarda mas en correr (mucho mas historico, sobre todo en 15m).

Requisitos:
    pip install requests pandas numpy

Uso:
    python3 sol_vwap_intraday_backtest_v3.py
"""

import time
import requests
import pandas as pd
import numpy as np

SYMBOL = "SOLUSDT"
BASE_URL = "https://api.binance.com/api/v3/klines"

INTERVALS_TO_TEST = ["15m", "1h", "4h"]
START_DATE = "2020-08-11"           # listado aproximado de SOLUSDT en Binance
TRAIN_TEST_SPLIT = "2024-06-01"     # mismo corte que usamos en el swing

VOL_AVG_WINDOW = 20
TARGET_PCT = 0.02
STOP_BELOW_VWAP_PCT = 0.01
FEE_PCT = 0.001

# Dos variantes a comparar por timeframe: con filtro de volumen y sin el
VOL_FILTER_VARIANTS = {
    "con_filtro_vol": 1.2,   # requiere volumen >= 1.2x su promedio movil
    "sin_filtro_vol": 0.0,   # desactivado (cualquier volumen cuenta)
}

MAX_HOLD_BARS_MAP = {"15m": 32, "1h": 12, "4h": 6}


def fetch_all_klines(symbol, interval, start_date):
    start_time = int(pd.Timestamp(start_date).timestamp() * 1000)
    end_time = int(time.time() * 1000)
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
        time.sleep(0.2)
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


def add_daily_bias(intraday_df, daily_df):
    daily_df = daily_df.copy()
    daily_df["date"] = daily_df["open_time"].dt.date
    daily_df["bias_ok"] = (daily_df["close"] > daily_df["ema_slow"]) & (daily_df["ema_fast"] > daily_df["ema_slow"])
    bias_map = daily_df.set_index("date")["bias_ok"].to_dict()

    intraday_df = intraday_df.copy()
    intraday_df["date"] = intraday_df["open_time"].dt.date
    intraday_df["daily_bias_ok"] = intraday_df["date"].map(bias_map).fillna(False)
    return intraday_df


def add_session_vwap(df):
    df = df.copy()
    df["date"] = df["open_time"].dt.date
    typical = (df["high"] + df["low"] + df["close"]) / 3
    df["tp_vol"] = typical * df["volume"]
    df["cum_tp_vol"] = df.groupby("date")["tp_vol"].cumsum()
    df["cum_vol"] = df.groupby("date")["volume"].cumsum()
    df["vwap"] = df["cum_tp_vol"] / df["cum_vol"]
    return df


def backtest_vwap_reclaim(df, max_hold_bars, vol_confirm_ratio):
    df = df.copy()
    df["vol_avg"] = df["volume"].rolling(VOL_AVG_WINDOW).mean()

    trades = []
    in_pos = False
    entry_price = entry_idx = None

    for i in range(VOL_AVG_WINDOW + 1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]

        if not in_pos:
            reclaimed = prev["close"] < prev["vwap"] and row["close"] >= row["vwap"]
            if vol_confirm_ratio > 0:
                vol_ok = not pd.isna(row["vol_avg"]) and row["vol_avg"] > 0 and (row["volume"] / row["vol_avg"]) >= vol_confirm_ratio
            else:
                vol_ok = True  # filtro desactivado
            if reclaimed and vol_ok and row["daily_bias_ok"]:
                in_pos = True
                entry_price = row["close"]
                entry_idx = i
        else:
            bars_held = i - entry_idx
            target_price = entry_price * (1 + TARGET_PCT)
            invalid_price = row["vwap"] * (1 - STOP_BELOW_VWAP_PCT)

            if row["close"] >= target_price:
                pnl = (target_price - entry_price) / entry_price * 100 - 2*FEE_PCT*100
                trades.append({"entry_time": df.iloc[entry_idx]["open_time"], "exit_time": row["open_time"],
                                "pnl_pct": pnl, "reason": "target"})
                in_pos = False
            elif row["close"] <= invalid_price:
                pnl = (invalid_price - entry_price) / entry_price * 100 - 2*FEE_PCT*100
                trades.append({"entry_time": df.iloc[entry_idx]["open_time"], "exit_time": row["open_time"],
                                "pnl_pct": pnl, "reason": "invalidacion_vwap"})
                in_pos = False
            elif bars_held >= max_hold_bars:
                pnl = (row["close"] - entry_price) / entry_price * 100 - 2*FEE_PCT*100
                trades.append({"entry_time": df.iloc[entry_idx]["open_time"], "exit_time": row["open_time"],
                                "pnl_pct": pnl, "reason": "timeout"})
                in_pos = False

    return pd.DataFrame(trades)


def stats(trades_df):
    if trades_df.empty:
        return dict(n=0, winrate=None, expectancy=None)
    wins = trades_df[trades_df.pnl_pct > 0]
    wr = len(wins) / len(trades_df) * 100
    exp = trades_df.pnl_pct.mean()
    return dict(n=len(trades_df), winrate=round(wr, 1), expectancy=round(exp, 3))


if __name__ == "__main__":
    print(f"Descargando historico diario completo de {SYMBOL} desde {START_DATE} (para bias)...")
    daily_df = fetch_all_klines(SYMBOL, "1d", START_DATE)
    daily_df = add_ema(daily_df, 50, "ema_fast")
    daily_df = add_ema(daily_df, 200, "ema_slow")
    print(f"{len(daily_df)} velas diarias descargadas")

    split_date = pd.Timestamp(TRAIN_TEST_SPLIT)
    results = []

    for interval in INTERVALS_TO_TEST:
        print(f"\n{'='*70}\nDescargando historico COMPLETO {interval} de {SYMBOL} desde {START_DATE}...")
        print("(esto puede tardar varios minutos en 15m, son muchas velas)")
        df = fetch_all_klines(SYMBOL, interval, START_DATE)
        print(f"{len(df)} velas {interval} descargadas ({df['open_time'].min().date()} a {df['open_time'].max().date()})")

        df = add_session_vwap(df)
        df = add_daily_bias(df, daily_df)

        df_train = df[df["open_time"] < split_date].reset_index(drop=True)
        df_test = df[df["open_time"] >= split_date].reset_index(drop=True)
        max_hold = MAX_HOLD_BARS_MAP[interval]

        for variant_name, vol_ratio in VOL_FILTER_VARIANTS.items():
            tr_trades = backtest_vwap_reclaim(df_train, max_hold, vol_ratio)
            te_trades = backtest_vwap_reclaim(df_test, max_hold, vol_ratio)
            tr_stats = stats(tr_trades)
            te_stats = stats(te_trades)

            print(f"\n--- {interval} | {variant_name} ---")
            print(f"TRAIN: n={tr_stats['n']} winrate={tr_stats['winrate']} expectancy={tr_stats['expectancy']}")
            print(f"TEST:  n={te_stats['n']} winrate={te_stats['winrate']} expectancy={te_stats['expectancy']}")

            results.append({
                "interval": interval, "variant": variant_name,
                "train_n": tr_stats["n"], "train_wr": tr_stats["winrate"], "train_exp": tr_stats["expectancy"],
                "test_n": te_stats["n"], "test_wr": te_stats["winrate"], "test_exp": te_stats["expectancy"],
            })

    summary_df = pd.DataFrame(results)
    pd.set_option("display.width", 140)
    print(f"\n{'='*70}\nRESUMEN COMPLETO — historico total + train/test + con/sin filtro volumen\n{'='*70}")
    print(summary_df.to_string(index=False))
    summary_df.to_csv("sol_vwap_summary_full.csv", index=False)
    print("\nGuardado: sol_vwap_summary_full.csv")
