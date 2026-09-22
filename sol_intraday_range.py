"""
SOL Intradia - Range Trading dentro del Volume Profile
========================================================
Hipotesis a probar: comprar cuando el precio toca cerca del VAL (parte baja
del rango de valor) y vender cuando toca cerca del VAH (parte alta), dentro
de una ventana de N horas/dias, usando velas 1h.

IMPORTANTE: esto asume que el dia respeta el rango (reversion). Si el dia
es de tendencia/breakout, este approach compra caidas que siguen cayendo.
El script mide exactamente cuantas veces pasa cada cosa -- no lo asumas,
mira el resultado.

Incluye fees de Binance spot (0.1% taker por lado, 0.2% redondo) para que
el numero final sea realista y no una fantasia sin costos de transaccion.

Requisitos:
    pip install requests pandas numpy

Uso:
    python3 sol_intraday_range.py
"""

import time
import requests
import pandas as pd
import numpy as np

SYMBOL = "SOLUSDT"
INTERVAL = "1h"
BASE_URL = "https://api.binance.com/api/v3/klines"

LOOKBACK_DAYS = 180          # ~6 meses de velas 1h (suficiente volumen de datos intradia)
VP_LOOKBACK_DAYS = 5         # ventana de dias previos para calcular el Value Area de referencia
VP_BINS = 20

ENTRY_TOLERANCE_PCT = 0.005  # que tan cerca del VAL cuenta como "toco el VAL" (0.5%)
EXIT_TOLERANCE_PCT = 0.005   # igual para el VAH
BREAKDOWN_STOP_PCT = 0.02    # si rompe 2% bajo el VAL sin revertir, se asume breakout -> stop
MAX_HOLD_HOURS = 24          # no dejar la posicion abierta mas de 1 dia
FEE_PCT = 0.001              # 0.1% por lado (ajusta si tienes VIP/BNB discount)


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


def daily_value_area(day_df, bins=VP_BINS):
    """Volume profile de un bloque de velas 1h (VP_LOOKBACK_DAYS dias previos)."""
    lo, hi = day_df["low"].min(), day_df["high"].max()
    if hi == lo:
        return None, None, None
    edges = np.linspace(lo, hi, bins + 1)
    vol_bin = np.zeros(bins)
    for _, r in day_df.iterrows():
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
    return poc, vah, val


def backtest_intraday_range(df):
    bars_per_day = 24
    vp_window_bars = VP_LOOKBACK_DAYS * bars_per_day

    trades = []
    breakout_days_skipped = 0
    no_touch_days = 0
    in_pos = False
    entry_price = entry_time = val_ref = vah_ref = None

    i = vp_window_bars
    while i < len(df):
        # Recalcula el Value Area de referencia 1 vez por dia (cada 24 barras 1h)
        ref_window = df.iloc[i - vp_window_bars:i]
        poc, vah, val = daily_value_area(ref_window)
        if val is None:
            i += bars_per_day
            continue

        day_slice = df.iloc[i:i + bars_per_day]
        touched_val = False
        touched_vah_after_entry = False

        for _, bar in day_slice.iterrows():
            price_low, price_high, price_close = bar["low"], bar["high"], bar["close"]

            if not in_pos:
                # se considera "toco el VAL" si el low de la vela entro en la tolerancia
                if price_low <= val * (1 + ENTRY_TOLERANCE_PCT):
                    in_pos = True
                    entry_price = val  # asumimos fill en el nivel (optimista pero simple)
                    entry_time = bar["open_time"]
                    val_ref, vah_ref = val, vah
                    touched_val = True
            else:
                hold_hours = (bar["open_time"] - entry_time).total_seconds() / 3600
                stop_level = val_ref * (1 - BREAKDOWN_STOP_PCT)

                if price_high >= vah_ref * (1 - EXIT_TOLERANCE_PCT):
                    # vendio cerca del VAH -> trade ganador
                    exit_price = vah_ref
                    gross = (exit_price - entry_price) / entry_price
                    net = gross - 2 * FEE_PCT
                    trades.append({"entry_time": entry_time, "exit_time": bar["open_time"],
                                    "pnl_pct": net * 100, "reason": "target_vah"})
                    in_pos = False
                elif price_low <= stop_level:
                    # rompio el VAL con fuerza -> dia de breakout, no de rango. Se corta.
                    exit_price = stop_level
                    gross = (exit_price - entry_price) / entry_price
                    net = gross - 2 * FEE_PCT
                    trades.append({"entry_time": entry_time, "exit_time": bar["open_time"],
                                    "pnl_pct": net * 100, "reason": "breakdown_stop"})
                    in_pos = False
                    breakout_days_skipped += 1
                elif hold_hours >= MAX_HOLD_HOURS:
                    # se acabo el tiempo, sale al close
                    exit_price = price_close
                    gross = (exit_price - entry_price) / entry_price
                    net = gross - 2 * FEE_PCT
                    trades.append({"entry_time": entry_time, "exit_time": bar["open_time"],
                                    "pnl_pct": net * 100, "reason": "timeout"})
                    in_pos = False

        if not touched_val and not in_pos:
            no_touch_days += 1

        i += bars_per_day

    return pd.DataFrame(trades), breakout_days_skipped, no_touch_days


def summarize(trades, breakout_skipped, no_touch_days, total_days):
    print(f"\n{'='*55}")
    print(f"BACKTEST INTRADIA - Range trading VAL->VAH")
    print(f"{'='*55}")
    print(f"Dias analizados:              {total_days}")
    print(f"Dias sin tocar el VAL:        {no_touch_days} ({no_touch_days/total_days*100:.1f}%)")
    print(f"Trades cortados por breakout: {breakout_skipped}")

    if trades.empty:
        print("\nNo se generaron trades completos.")
        return

    wins = trades[trades.pnl_pct > 0]
    winrate = len(wins) / len(trades) * 100
    avg_win = wins.pnl_pct.mean() if not wins.empty else 0
    losses = trades[trades.pnl_pct <= 0]
    avg_loss = losses.pnl_pct.mean() if not losses.empty else 0
    expectancy = trades.pnl_pct.mean()

    equity = 100
    for p in trades.pnl_pct:
        equity *= (1 + p/100)

    print(f"\nTotal trades:         {len(trades)}")
    print(f"Winrate:              {winrate:.1f}%")
    print(f"Avg win / Avg loss:   {avg_win:.2f}% / {avg_loss:.2f}%")
    print(f"Expectancy (con fees):{expectancy:.3f}%  <- ya descontado 0.2% de fee redondo")
    print(f"Equity final (100->): {equity:.2f}")
    print(f"\nDistribucion por razon de salida:")
    print(trades["reason"].value_counts())


if __name__ == "__main__":
    print(f"Descargando velas 1h de {SYMBOL} (~{LOOKBACK_DAYS} dias)...")
    df = fetch_klines()
    print(f"{len(df)} velas 1h descargadas")

    trades, breakout_skipped, no_touch = backtest_intraday_range(df)
    total_days = len(df) // 24
    summarize(trades, breakout_skipped, no_touch, total_days)

    trades.to_csv("sol_intraday_trades.csv", index=False)
    print("\nGuardado: sol_intraday_trades.csv")
