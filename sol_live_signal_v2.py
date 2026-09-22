"""
SOL Swing - Señal en Vivo CON ESTADO (v2)
============================================
Igual que sol_live_signal.py, pero con memoria entre corridas:

- Si NO tienes posición abierta: busca setup nuevo (igual que antes) y,
  si hay uno, te pregunta si entraste y a qué precio. Guarda ese dato.
- Si YA tienes posición abierta (segun el archivo de estado): te pregunta
  si ya la cerraste. Si no, evalua el precio actual contra tu stop/target
  fijados al entrar, mas RSI y tendencia vigentes, y te dice si conviene
  MANTENER o si ya hay razon para SALIR.

Guarda:
- sol_position_state.json  -> el estado actual (abierta o no, y sus datos)
- sol_trade_log.csv        -> historial de operaciones ya cerradas

Requisitos:
    pip install requests pandas numpy

Uso:
    python3 sol_live_signal_v2.py
"""

import json
import os
import time
from datetime import datetime

import requests
import pandas as pd
import numpy as np

SYMBOL = "SOLUSDT"
INTERVAL = "1d"
BASE_URL = "https://api.binance.com/api/v3/klines"

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
LOOKBACK_DAYS = 400

STATE_FILE = "sol_position_state.json"
LOG_FILE = "sol_trade_log.csv"


# -------------------- DATOS / INDICADORES (igual que v1) --------------------
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


# -------------------- ESTADO --------------------
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"in_position": False}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def log_closed_trade(state, exit_price, exit_reason):
    entry_price = state["entry_price"]
    pnl_pct = (exit_price - entry_price) / entry_price * 100
    row = {
        "entry_date": state["entry_date"],
        "entry_price": entry_price,
        "exit_date": datetime.now().strftime("%Y-%m-%d"),
        "exit_price": exit_price,
        "pnl_pct": round(pnl_pct, 2),
        "exit_reason": exit_reason,
    }
    df_row = pd.DataFrame([row])
    if os.path.exists(LOG_FILE):
        df_row.to_csv(LOG_FILE, mode="a", header=False, index=False)
    else:
        df_row.to_csv(LOG_FILE, mode="w", header=True, index=False)
    print(f"\nTrade guardado en {LOG_FILE}: PnL = {pnl_pct:.2f}%")


def ask_yes_no(prompt):
    while True:
        resp = input(prompt + " (s/n): ").strip().lower()
        if resp in ("s", "si", "sí", "y", "yes"):
            return True
        if resp in ("n", "no"):
            return False
        print("Responde 's' o 'n'.")


def ask_float(prompt, default=None):
    while True:
        raw = input(prompt).strip()
        if raw == "" and default is not None:
            return default
        try:
            return float(raw)
        except ValueError:
            print("Ingresa un número válido.")


# -------------------- FLUJO PRINCIPAL --------------------
def check_new_setup(df, poc, vah, val, hvns):
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
    print(f"{'-'*55}")
    print(f"Bias alcista:                {'SI' if bias_ok else 'NO'}")
    print(f"RSI en zona de entrada 45-60:{'SI' if rsi_reset else 'NO'}")
    print(f"Precio cerca de HVN/POC:     {'SI' if near_hvn else 'NO'}")
    print(f"{'='*55}")

    if not setup_active:
        faltantes = []
        if not bias_ok: faltantes.append("bias alcista")
        if not rsi_reset: faltantes.append("RSI en zona 45-60")
        if not near_hvn: faltantes.append("proximidad a HVN/POC")
        print(f">>> SIN SETUP — falta: {', '.join(faltantes)} <<<")
        return

    stop_level = poc * (1 - STOP_PCT)
    print(">>> SETUP ACTIVO — condiciones de compra cumplidas <<<")
    print(f"Referencia de entrada: {price:.2f}")
    print(f"Stop sugerido:         {stop_level:.2f}  (-{STOP_PCT*100:.0f}% del POC)")
    print(f"Target (VAH):          {vah:.2f}")

    entered = ask_yes_no("\n¿Entraste a esta operación?")
    if entered:
        entry_price = ask_float(f"¿A qué precio entraste? (Enter = usar {price:.2f}): ", default=price)
        state = {
            "in_position": True,
            "entry_date": last["open_time"].strftime("%Y-%m-%d"),
            "entry_price": entry_price,
            "poc_at_entry": poc,
            "stop_level": entry_price and (poc * (1 - STOP_PCT)),
            "target_vah": vah,
        }
        save_state(state)
        print(f"\nGuardado. Posición abierta desde {state['entry_date']} a {entry_price:.2f}.")
        print(f"Stop: {state['stop_level']:.2f} | Target: {state['target_vah']:.2f}")
    else:
        print("\nOk, no se guardó ninguna posición. Vuelve a correr el script cuando quieras.")


def evaluate_open_position(state, df):
    last = df.iloc[-1]
    price = last["close"]

    print(f"\n{'='*55}")
    print(f"POSICIÓN ABIERTA desde {state['entry_date']} a {state['entry_price']:.2f}")
    print(f"{'='*55}")

    ya_cerraste = ask_yes_no("¿Ya cerraste esta operación?")
    if ya_cerraste:
        exit_price = ask_float(f"¿A qué precio cerraste? (Enter = usar precio actual {price:.2f}): ", default=price)
        exit_reason = input("Motivo de salida (target/stop/rsi/tendencia/otro): ").strip() or "manual"
        log_closed_trade(state, exit_price, exit_reason)
        save_state({"in_position": False})
        print("\nEstado reiniciado — la próxima corrida buscará un setup nuevo.")
        return

    # Sigue abierta: evaluar
    entry_price = state["entry_price"]
    stop_level = state["stop_level"]
    target = state["target_vah"]
    days_held = (last["open_time"] - pd.Timestamp(state["entry_date"])).days
    unrealized_pnl = (price - entry_price) / entry_price * 100

    target_hit = price >= target
    stop_hit = price <= stop_level
    rsi_ob = last["rsi"] >= RSI_EXIT_OVERBOUGHT
    trend_broken = last["ema_fast"] < last["ema_slow"]

    dist_to_target = (target - price) / price * 100
    dist_to_stop = (price - stop_level) / price * 100

    print(f"\nPrecio actual:          {price:.2f}")
    print(f"Días en posición:       {days_held}")
    print(f"PnL no realizado:       {unrealized_pnl:+.2f}%")
    print(f"Distancia al target:    {dist_to_target:+.2f}%  (target={target:.2f})")
    print(f"Distancia al stop:      {dist_to_stop:+.2f}%  (stop={stop_level:.2f})")
    print(f"RSI(14) actual:         {last['rsi']:.2f}")
    print(f"EMA50 / EMA200:         {last['ema_fast']:.2f} / {last['ema_slow']:.2f}")
    print(f"{'-'*55}")

    razones_salida = []
    if target_hit: razones_salida.append("precio alcanzó el target (VAH)")
    if stop_hit: razones_salida.append("precio rompió el stop de invalidación")
    if rsi_ob: razones_salida.append(f"RSI en sobrecompra ({last['rsi']:.1f} >= {RSI_EXIT_OVERBOUGHT})")
    if trend_broken: razones_salida.append("la tendencia se rompió (EMA50 < EMA200)")

    if razones_salida:
        print(">>> HAY RAZÓN PARA CONSIDERAR SALIR <<<")
        for r in razones_salida:
            print(f"  - {r}")
    else:
        print(">>> MANTENER — ninguna condición de salida se cumple todavía <<<")

    print(f"{'='*55}")
    print("(Recuerda: la salida final la ejecutas tú manualmente en Binance.")
    print(" La próxima vez que corras el script, te preguntará de nuevo si ya cerraste.)")


if __name__ == "__main__":
    print(f"Descargando datos recientes de {SYMBOL}...")
    df = fetch_klines()
    df = add_ema(df, EMA_FAST, "ema_fast")
    df = add_ema(df, EMA_SLOW, "ema_slow")
    df = add_rsi(df)

    state = load_state()

    if state.get("in_position"):
        evaluate_open_position(state, df)
    else:
        poc, vah, val, hvns = volume_profile_latest(df)
        check_new_setup(df, poc, vah, val, hvns)
