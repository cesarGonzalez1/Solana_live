# SOL Live Signal — Guía de uso diario

Script: `sol_live_signal.py`
Estrategia: Swing SOL/USDT spot, solo largos
Parámetros validados: VP window=90d, RSI entrada 45-60, stop=6% del POC, target=VAH

---

## 1. Instalación (una sola vez)

```bash
cd ~/ruta/donde/guardaste/el/script
python3 -m venv venv
source venv/bin/activate
pip install requests pandas numpy
```

## 2. Correrlo

Cada vez que quieras revisar:

```bash
source venv/bin/activate     # si no está ya activado
python3 sol_live_signal.py
```

## 3. Cuándo correrlo

La estrategia usa **velas diarias**, así que correrlo más de una vez al día no cambia nada — la vela del día no cierra hasta las 00:00 UTC (18:00 hora Ciudad de México).

- **Si no tienes trade abierto:** corre el script 1 vez al día, después del cierre de la vela diaria (después de las 18:00 hora CDMX es lo más seguro para ver el dato ya consolidado). Si dice `SETUP ACTIVO`, ese es tu día de entrada.
- **Si ya tienes una posición abierta:** sigue corriéndolo 1 vez al día. El script no monitorea automáticamente tu posición — pero cada corrida te muestra los mismos datos (precio, RSI, EMA50/200, VAH) que necesitas para decidir si toca salir. Usa esta checklist manual mientras estás dentro:

| Condición de salida | Cómo verlo en el output |
|---|---|
| **Target alcanzado** | Precio actual ≥ `VAH` impreso |
| **Sobrecompra** | `RSI(14)` ≥ 75 |
| **Cambio de tendencia** | `EMA50` cruza por debajo de `EMA200` |
| **Stop de invalidación** | Precio cae por debajo del nivel de stop que anotaste al entrar (POC del día de entrada × 0.94 — el script te lo imprime en ese momento, guárdalo) |

Cualquiera de las 4 que se cumpla primero = sales, sin esperar a que se cumplan todas.

## 4. Qué hacer con el output

**Cuando dice `SETUP ACTIVO`:**
1. Anota el precio de referencia, el stop sugerido y el target (VAH) que imprime.
2. Ejecuta la compra manual en Binance al precio de mercado (o con una orden límite cerca del nivel).
3. Guarda esos 3 números (entrada, stop, target) en tu journal de Notion — mismo hábito que ya tienes con 6A.

**Cuando dice `SIN SETUP`:**
- Te dice explícitamente qué condición falta (`bias alcista`, `RSI en zona 45-60`, o `proximidad a HVN/POC`). No hagas nada, solo vuelve a correrlo al día siguiente.

## 5. Automatizar el recordatorio (opcional)

Si quieres que te llegue el output todos los días sin acordarte de correrlo a mano, un cron job simple en Mac:

```bash
crontab -e
```

Agrega esta línea (corre todos los días a las 19:00 hora CDMX y guarda el resultado en un log):

```
0 19 * * * cd /ruta/completa/al/script && /ruta/completa/al/script/venv/bin/python3 sol_live_signal.py >> sol_signal_log.txt 2>&1
```

Esto **no ejecuta órdenes**, solo te deja el reporte guardado en `sol_signal_log.txt` para que lo revises cuando quieras — la decisión de comprar sigue siendo manual, tal como querías (solo comprar/vender, sin automatizar la ejecución).

## 6. Registro recomendado en tu journal

Cada vez que el script marque `SETUP ACTIVO` y decidas (o no) tomarlo, anota en Notion:
- Fecha, precio de entrada, RSI y POC/VAH/VAL de ese día
- Si tomaste el trade o no, y por qué
- Resultado final (target / stop / timeout por cambio de tendencia)

Esto te va a dejar la misma base de datos que ya usas para refinar 6A — con suficientes trades reales podrás comparar el resultado en vivo contra el backtest (train n=12, test n=13, expectancy +2.45%) y confirmar si el edge se sostiene fuera del backtest también.
