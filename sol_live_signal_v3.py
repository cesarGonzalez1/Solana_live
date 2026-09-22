"""
SOL Swing - Señal en Vivo v3
=============================
Agrega sobre v2:
- Deteccion de volumen anomalo (posible catalizador/noticia) -> avisa revisar
  manualmente, el script NO lee noticias, solo detecta la anomalia en precio/volumen.
- Si el precio toca el target (VAH) pero el setup sigue tecnicamente fuerte
  (bias intacto, RSI no extremo, volumen sostenido), sugiere DEJAR CORRER
  con nuevo stop (breakeven+) y siguiente resistencia (HVN mas amplio),
  en vez de forzar salida solo porque tocó el primer target.

Guarda igual: sol_position_state.json, sol_trade_log.csv

Uso: python3 sol_live_signal_v3.py
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
VP_WINDOW_EXT = 180          # ventana ancha para buscar la SIGUIENTE resistencia tras el primer target
RSI_LEN = 14
RSI_ENTRY_LOW = 45
RSI_ENTRY_HIGH = 60
RSI_EXIT_OVERBOUGHT = 75
RSI_EXTENSION_MAX = 80       # arriba de esto, ya no se sugiere extender aunque lo demas se vea bien
STOP_PCT = 0.06
HVN_PROXIMITY_PCT = 0.015
EMA_FAST = 50
EMA_SLOW = 200
VP_BINS = 24
LOOKBACK_DAYS = 400
VOL_AVG_WINDOW = 20
VOL_SPIKE_RATIO = 1.5        # volumen hoy >= 1.5x promedio 20d = anomalia

STATE_FILE = "sol_position_state.json"
LOG_FILE = "sol_trade_log.csv"


# -------------------- DATOS / INDICADORES --------------------
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


def add_volume_avg(df, window=VOL_AVG_WINDOW):
    df["vol_avg"] = df["volume"].rolling(window).mean()
    return df


def compute_volume_profile(window_df, bins=VP_BINS):
    lo, hi = window_df["low"].min(), window_df["high"].max()
    if hi == lo:
        return None, None, None, []
    edges = np.linspace(lo, hi, bins + 1)
    vol_bin = np.zeros(bins)
    for _, r in window_df.iterrows():
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


def volume_profile_latest(df, window=VP_WINDOW):
    return compute_volume_profile(df.iloc[-window:])


def near_any(price, levels, pct=HVN_PROXIMITY_PCT):
    if not levels:
        return False
    return any(abs(price-l)/price <= pct for l in levels)


def find_next_resistance(df, current_price, window=VP_WINDOW_EXT):
    """Busca el HVN mas cercano POR ARRIBA del precio actual, usando ventana mas ancha."""
    _, _, _, hvns = compute_volume_profile(df.iloc[-window:])
    above = sorted([h for h in hvns if h > current_price * 1.01])
    return above[0] if above else None


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
        "entry_date": state["entry_date"], "entry_price": entry_price,
        "exit_date": datetime.now().strftime("%Y-%m-%d"), "exit_price": exit_price,
        "pnl_pct": round(pnl_pct, 2), "exit_reason": exit_reason,
    }
    df_row = pd.DataFrame([row])
    if os.path.exists(LOG_FILE):
        df_row.to_csv(LOG_FILE, mode="a", header=False, index=False)
    else:
        df_row.to_csv(LOG_FILE, mode="w", header=True, index=False)
    print(f"\nGuardado en {LOG_FILE}: PnL = {pnl_pct:.2f}%")


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
            print("Numero invalido.")


def check_volume_anomaly(df):
    last = df.iloc[-1]
    if pd.isna(last["vol_avg"]) or last["vol_avg"] == 0:
        return None
    ratio = last["volume"] / last["vol_avg"]
    if ratio >= VOL_SPIKE_RATIO:
        return ratio
    return None


# -------------------- FLUJO NUEVO SETUP --------------------
def check_new_setup(df, poc, vah, val, hvns):
    last = df.iloc[-1]
    price = last["close"]

    bias_ok = (price > last["ema_slow"]) and (last["ema_fast"] > last["ema_slow"])
    rsi_reset = RSI_ENTRY_LOW <= last["rsi"] <= RSI_ENTRY_HIGH
    near_hvn = near_any(price, hvns + [poc])
    setup_active = bias_ok and rsi_reset and near_hvn

    print(f"\n{'='*55}\nSOL SWING - SEÑAL AL {last['open_time'].date()}\n{'='*55}")
    print(f"Precio: {price:.2f} | EMA50/200: {last['ema_fast']:.2f}/{last['ema_slow']:.2f} | RSI: {last['rsi']:.2f}")
    print(f"POC/VAH/VAL: {poc:.2f}/{vah:.2f}/{val:.2f}")
    print(f"Bias:{'SI' if bias_ok else 'NO'}  RSI-zona:{'SI' if rsi_reset else 'NO'}  Cerca-HVN:{'SI' if near_hvn else 'NO'}")
    print("="*55)

    anomaly = check_volume_anomaly(df)
    if anomaly:
        print(f"⚠ VOLUMEN ANOMALO: hoy {anomaly:.1f}x el promedio 20d — posible catalizador/noticia. Revisa manualmente (Binance news, CoinDesk, X).")

    if not setup_active:
        faltantes = [n for cond, n in [(bias_ok,"bias alcista"),(rsi_reset,"RSI 45-60"),(near_hvn,"cerca HVN/POC")] if not cond]
        print(f">>> SIN SETUP — falta: {', '.join(faltantes)} <<<")
        return

    stop_level = poc * (1 - STOP_PCT)
    print(">>> SETUP ACTIVO <<<")
    print(f"Entrada ref: {price:.2f} | Stop: {stop_level:.2f} | Target: {vah:.2f}")

    if ask_yes_no("\n¿Entraste?"):
        entry_price = ask_float(f"Precio de entrada (Enter={price:.2f}): ", default=price)
        state = {
            "in_position": True, "entry_date": last["open_time"].strftime("%Y-%m-%d"),
            "entry_price": entry_price, "poc_at_entry": poc,
            "stop_level": poc * (1 - STOP_PCT), "target_vah": vah,
            "extended": False,
        }
        save_state(state)
        print(f"\nGuardado. Stop:{state['stop_level']:.2f} Target:{state['target_vah']:.2f}")
    else:
        print("\nOk, no se guardó nada.")


# -------------------- FLUJO POSICION ABIERTA --------------------
def evaluate_open_position(state, df):
    last = df.iloc[-1]
    price = last["close"]

    print(f"\n{'='*55}\nPOSICIÓN ABIERTA desde {state['entry_date']} a {state['entry_price']:.2f}\n{'='*55}")

    if ask_yes_no("¿Ya cerraste esta operación?"):
        exit_price = ask_float(f"Precio de cierre (Enter=actual {price:.2f}): ", default=price)
        reason = input("Motivo (target/stop/rsi/tendencia/otro): ").strip() or "manual"
        log_closed_trade(state, exit_price, reason)
        save_state({"in_position": False})
        print("\nEstado reiniciado.")
        return

    entry_price = state["entry_price"]
    stop_level = state["stop_level"]
    target = state["target_vah"]
    days_held = (last["open_time"] - pd.Timestamp(state["entry_date"])).days
    unrealized_pnl = (price - entry_price) / entry_price * 100

    target_hit = price >= target
    stop_hit = price <= stop_level
    rsi_ob = last["rsi"] >= RSI_EXIT_OVERBOUGHT
    trend_broken = last["ema_fast"] < last["ema_slow"]

    print(f"\nPrecio: {price:.2f} | Días: {days_held} | PnL: {unrealized_pnl:+.2f}%")
    print(f"Target: {target:.2f} ({(target-price)/price*100:+.2f}%) | Stop: {stop_level:.2f} ({(price-stop_level)/price*100:+.2f}%)")
    print(f"RSI: {last['rsi']:.2f} | EMA50/200: {last['ema_fast']:.2f}/{last['ema_slow']:.2f}")

    anomaly = check_volume_anomaly(df)
    if anomaly:
        print(f"⚠ VOLUMEN ANOMALO: {anomaly:.1f}x promedio 20d — algo esta moviendo el precio fuera de lo normal. Revisa noticias manualmente antes de decidir.")

    print("-"*55)

    # --- Caso: tocó target pero setup sigue fuerte -> evaluar extension ---
    if target_hit and not stop_hit:
        bias_ok = last["ema_fast"] > last["ema_slow"]
        rsi_ok_extend = last["rsi"] < RSI_EXTENSION_MAX
        vol_sostenido = anomaly is not None or (last["volume"] >= last["vol_avg"] if not pd.isna(last["vol_avg"]) else False)

        if bias_ok and rsi_ok_extend:
            next_target = find_next_resistance(df, price)
            new_stop = max(stop_level, entry_price * 1.02)  # breakeven + 2% de colchon

            print(">>> TARGET ALCANZADO — pero setup TECNICAMENTE aun fuerte <<<")
            print(f"  Bias intacto: SI | RSI: {last['rsi']:.1f} (<{RSI_EXTENSION_MAX} limite extension)")
            print(f"  Volumen {'sostenido/alto' if vol_sostenido else 'normal'}")
            if next_target:
                print(f"\n  OPCION: dejar correr. Nuevo stop sugerido (breakeven+): {new_stop:.2f}")
                print(f"  Siguiente resistencia (HVN mas amplio): {next_target:.2f}")
                print(f"  O tomar ganancia parcial aqui y dejar el resto corriendo con el nuevo stop.")
            else:
                print(f"\n  No se encontró resistencia clara mas arriba en ventana de {VP_WINDOW_EXT}d.")
                print(f"  Considera tomar la ganancia completa aqui, o parcial con trailing stop en {new_stop:.2f}.")
            print(f"\n  Esto es lectura tecnica, no sabe la causa (noticia/evento). Si detectaste volumen anomalo arriba,")
            print(f"  ve que dice el mercado antes de decidir extender.")
        else:
            razon = "RSI extendido" if not rsi_ok_extend else "bias debilitandose"
            print(f">>> TARGET ALCANZADO — y setup ya no se ve fuerte ({razon}) <<<")
            print(f"  Recomendacion: tomar ganancia aqui, no extender.")
        print("="*55)
        return

    # --- Casos normales de salida ---
    razones = []
    if stop_hit: razones.append("rompió el stop")
    if rsi_ob: razones.append(f"RSI sobrecomprado ({last['rsi']:.1f})")
    if trend_broken: razones.append("tendencia rota (EMA50<EMA200)")

    if razones:
        print(">>> HAY RAZÓN PARA CONSIDERAR SALIR <<<")
        for r in razones: print(f"  - {r}")
    else:
        print(">>> MANTENER — ninguna condición de salida se cumple <<<")
    print("="*55)


if __name__ == "__main__":
    print(f"Descargando datos de {SYMBOL}...")
    df = fetch_klines()
    df = add_ema(df, EMA_FAST, "ema_fast")
    df = add_ema(df, EMA_SLOW, "ema_slow")
    df = add_rsi(df)
    df = add_volume_avg(df)

    state = load_state()
    if state.get("in_position"):
        evaluate_open_position(state, df)
    else:
        poc, vah, val, hvns = volume_profile_latest(df)
        check_new_setup(df, poc, vah, val, hvns)
