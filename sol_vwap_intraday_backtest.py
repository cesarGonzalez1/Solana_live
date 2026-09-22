"""
SOL Intradia v2 - VWAP Reclaim en multiples timeframes
=========================================================
Hipotesis (de la investigacion en foros/fuentes institucionales):
precio cae bajo el VWAP de sesion, luego lo "reclama" (cierra de nuevo
arriba) con volumen de confirmacion, DENTRO de un dia con bias diario
alcista (EMA50>EMA200 diario) -- eso es la entrada. Salida por target,
por perder el VWAP de nuevo (invalidacion), o por tiempo.

Prueba esto en 3 resoluciones (15m, 1h, 4h) para comparar, en vez de
asumir que una sola funciona. Incluye fees.

Requisitos:
    pip install requests pandas numpy

Uso:
    python3 sol_vwap_intraday_backtest.py
"""

import time
import requests
import pandas as pd
import numpy as np

SYMBOL = "SOLUSDT"
BASE_URL = "https://api.binance.com/api/v3/klines"

INTERVALS_TO_TEST = ["15m", "1h", "4h"]
LOOKBACK_DAYS = {"15m": 90, "1h": 180, "4h": 365}   # mas historia en TFs mas altos
DAILY_LOOKBACK_DAYS = 400  # para el filtro de bias diario (EMA50/200)

VOL_CONFIRM_RATIO = 1.2      # volumen de la vela de reclaim >= 1.2x su propio promedio movil
VOL_AVG_WINDOW = 20
TARGET_PCT = 0.02            # +2% desde el VWAP de reclaim
STOP_BELOW_VWAP_PCT = 0.01   # si vuelve a cerrar 1% bajo el VWAP, invalidado
MAX_HOLD_BARS = 12           # limite de velas en posicion (se ajusta al peso del TF)
FEE_PCT = 0.001              # 0.1% por lado


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


def add_daily_bias(intraday_df, daily_df):
    """Pega a cada vela intradia el bias del DIA correspondiente (EMA50>EMA200 diario)."""
    daily_df = daily_df.copy()
    daily_df["date"] = daily_df["open_time"].dt.date
    daily_df["bias_ok"] = (daily_df["close"] > daily_df["ema_slow"]) & (daily_df["ema_fast"] > daily_df["ema_slow"])
    bias_map = daily_df.set_index("date")["bias_ok"].to_dict()

    intraday_df = intraday_df.copy()
    intraday_df["date"] = intraday_df["open_time"].dt.date
    intraday_df["daily_bias_ok"] = intraday_df["date"].map(bias_map).fillna(False)
    return intraday_df


def add_session_vwap(df):
    """VWAP que resetea cada dia UTC (sesion = dia calendario)."""
    df = df.copy()
    df["date"] = df["open_time"].dt.date
    typical = (df["high"] + df["low"] + df["close"]) / 3
    df["tp_vol"] = typical * df["volume"]
    df["cum_tp_vol"] = df.groupby("date")["tp_vol"].cumsum()
    df["cum_vol"] = df.groupby("date")["volume"].cumsum()
    df["vwap"] = df["cum_tp_vol"] / df["cum_vol"]
    return df


def backtest_vwap_reclaim(df, max_hold_bars):
    df["vol_avg"] = df["volume"].rolling(VOL_AVG_WINDOW).mean()

    trades = []
    in_pos = False
    entry_price = entry_idx = None

    for i in range(VOL_AVG_WINDOW + 1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]

        if not in_pos:
            reclaimed = prev["close"] < prev["vwap"] and row["close"] >= row["vwap"]
            vol_ok = not pd.isna(row["vol_avg"]) and row["vol_avg"] > 0 and (row["volume"] / row["vol_avg"]) >= VOL_CONFIRM_RATIO
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


def summarize(interval, trades):
    print(f"\n{'='*55}")
    print(f"VWAP RECLAIM - timeframe {interval}")
    print(f"{'='*55}")
    if trades.empty:
        print("Sin trades generados.")
        return None

    wins = trades[trades.pnl_pct > 0]
    winrate = len(wins)/len(trades)*100
    exp = trades.pnl_pct.mean()
    equity = 100
    for p in trades.pnl_pct:
        equity *= (1 + p/100)

    print(f"Trades: {len(trades)} | Winrate: {winrate:.1f}% | Expectancy: {exp:.3f}%")
    print(f"Equity final (100->): {equity:.2f}")
    print(trades["reason"].value_counts())
    return {"interval": interval, "n_trades": len(trades), "winrate": round(winrate,1),
            "expectancy": round(exp,3), "equity_final": round(equity,2)}


if __name__ == "__main__":
    print(f"Descargando velas diarias de {SYMBOL} para el filtro de bias...")
    daily_df = fetch_klines(SYMBOL, "1d", DAILY_LOOKBACK_DAYS)
    daily_df = add_ema(daily_df, 50, "ema_fast")
    daily_df = add_ema(daily_df, 200, "ema_slow")

    max_hold_map = {"15m": 32, "1h": 12, "4h": 6}   # ~8h, ~12h, ~24h de espera max aprox

    results = []
    for interval in INTERVALS_TO_TEST:
        print(f"\nDescargando velas {interval} de {SYMBOL} (~{LOOKBACK_DAYS[interval]} dias)...")
        df = fetch_klines(SYMBOL, interval, LOOKBACK_DAYS[interval])
        print(f"{len(df)} velas descargadas")

        df = add_session_vwap(df)
        df = add_daily_bias(df, daily_df)

        trades = backtest_vwap_reclaim(df, max_hold_map[interval])
        stats = summarize(interval, trades)
        if stats:
            results.append(stats)

        trades.to_csv(f"sol_vwap_trades_{interval}.csv", index=False)

    if results:
        summary_df = pd.DataFrame(results)
        print(f"\n{'='*55}")
        print("RESUMEN COMPARATIVO POR TIMEFRAME")
        print(f"{'='*55}")
        print(summary_df.to_string(index=False))
        summary_df.to_csv("sol_vwap_summary_by_timeframe.csv", index=False)
        print("\nGuardado: sol_vwap_summary_by_timeframe.csv + un CSV de trades por timeframe")
