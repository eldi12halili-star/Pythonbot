#!/usr/bin/env python3
"""
4H trend-following swing trading strategy + backtester.

Strategy (all decisions on CLOSED 4h candles, orders filled at the NEXT candle's open):
  Trend filter : EMA(fast) > EMA(slow) and close > EMA(slow)      (long)
                 EMA(fast) < EMA(slow) and close < EMA(slow)      (short, optional)
  Strength     : ADX(14) > adx_min  (only trade when a trend actually exists)
  Entry        : close breaks the highest high (long) / lowest low (short)
                 of the previous `breakout` candles (Donchian breakout)
  Initial stop : entry -/+ atr_stop * ATR(14)
  Trailing stop: chandelier stop = highest high since entry - atr_trail * ATR
                 (lowest low + atr_trail * ATR for shorts), only ever tightens
  Trend exit   : EMA(fast) crosses back through EMA(slow)
  Sizing       : risk `risk` fraction of equity per trade (distance to stop),
                 capped at `max_leverage` x equity notional
  Costs        : taker fee + slippage applied on every fill

Data: Binance public 4h klines (no API key needed), cached to ./data/, or any
CSV with columns time,open,high,low,close,volume via --csv.

Usage:
  python swing_trend_bot.py                       # BTCUSDT, long-only, since 2019
  python swing_trend_bot.py --symbol ETHUSDT --allow-short
  python swing_trend_bot.py --csv mydata.csv --plot
"""
from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

BINANCE_URLS = [
    "https://data-api.binance.vision/api/v3/klines",  # public market-data mirror
    "https://api.binance.com/api/v3/klines",
]
BARS_PER_YEAR = 6 * 365  # 4h candles


# --------------------------------------------------------------------------- data
def fetch_binance(symbol: str, interval: str, start: str, end: str | None = None) -> pd.DataFrame:
    import requests

    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000) if end else None
    rows, last_err = [], None
    for url in BINANCE_URLS:
        rows, cursor = [], start_ms
        try:
            while True:
                params = {"symbol": symbol, "interval": interval, "startTime": cursor, "limit": 1000}
                if end_ms:
                    params["endTime"] = end_ms
                r = requests.get(url, params=params, timeout=20)
                r.raise_for_status()
                batch = r.json()
                if not batch:
                    break
                rows.extend(batch)
                cursor = batch[-1][0] + 1
                if len(batch) < 1000:
                    break
                time.sleep(0.2)
            break
        except Exception as e:  # try next mirror
            last_err = e
            rows = []
    if not rows:
        raise RuntimeError(f"Could not download data from Binance: {last_err}")

    df = pd.DataFrame(rows).iloc[:, :6]
    df.columns = ["time", "open", "high", "low", "close", "volume"]
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df = df.set_index("time").astype(float)
    df = df[~df.index.duplicated()].sort_index()
    # drop the still-forming last candle
    bar = pd.Timedelta(interval)
    if df.index[-1] + bar > pd.Timestamp.now(tz="UTC"):
        df = df.iloc[:-1]
    return df


def load_data(args) -> pd.DataFrame:
    if args.csv:
        df = pd.read_csv(args.csv)
        df.columns = [c.lower() for c in df.columns]
        tcol = next(c for c in ("time", "timestamp", "date", "datetime", "open_time") if c in df.columns)
        t = df[tcol]
        df["time"] = pd.to_datetime(t, unit="ms", utc=True) if np.issubdtype(t.dtype, np.number) else pd.to_datetime(t, utc=True)
        df = df.set_index("time")[["open", "high", "low", "close"] + (["volume"] if "volume" in df.columns else [])]
        df = df.astype(float).sort_index()
        # resample to 4h in case a lower timeframe was supplied
        if len(df) > 2 and (df.index[1] - df.index[0]) < pd.Timedelta("4h"):
            agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
            if "volume" in df.columns:
                agg["volume"] = "sum"
            df = df.resample("4h").agg(agg).dropna()
        return df

    os.makedirs("data", exist_ok=True)
    cache = os.path.join("data", f"{args.symbol}_4h_{args.start}.csv")
    if os.path.exists(cache) and not args.refresh:
        df = pd.read_csv(cache, index_col="time", parse_dates=True)
    else:
        print(f"Downloading {args.symbol} 4h candles from Binance since {args.start} ...")
        df = fetch_binance(args.symbol, "4h", args.start, args.end)
        df.to_csv(cache)
    if args.end:
        df = df[df.index <= pd.Timestamp(args.end, tz="UTC")]
    return df


# --------------------------------------------------------------------- indicators
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    a = atr(df, n)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / a
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / a
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean()


@dataclass
class Params:
    ema_fast: int = 50
    ema_slow: int = 200
    breakout: int = 20
    adx_min: float = 20.0
    atr_len: int = 14
    atr_stop: float = 2.5
    atr_trail: float = 3.5
    risk: float = 0.01
    max_leverage: float = 1.0
    fee: float = 0.001        # 0.10% per side (Binance spot taker)
    slippage: float = 0.0005  # 0.05% per fill
    allow_short: bool = False


def add_signals(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    d = df.copy()
    d["ema_f"] = ema(d["close"], p.ema_fast)
    d["ema_s"] = ema(d["close"], p.ema_slow)
    d["atr"] = atr(d, p.atr_len)
    d["adx"] = adx(d, p.atr_len)
    d["hh"] = d["high"].rolling(p.breakout).max().shift(1)
    d["ll"] = d["low"].rolling(p.breakout).min().shift(1)
    up = (d["ema_f"] > d["ema_s"]) & (d["close"] > d["ema_s"])
    dn = (d["ema_f"] < d["ema_s"]) & (d["close"] < d["ema_s"])
    strong = d["adx"] > p.adx_min
    d["long_sig"] = up & strong & (d["close"] > d["hh"])
    d["short_sig"] = dn & strong & (d["close"] < d["ll"]) & p.allow_short
    d["trend_up"] = d["ema_f"] > d["ema_s"]
    d.iloc[: p.ema_slow, d.columns.get_loc("long_sig")] = False   # warm-up
    d.iloc[: p.ema_slow, d.columns.get_loc("short_sig")] = False
    return d


# ----------------------------------------------------------------------- backtest
@dataclass
class Trade:
    side: int
    entry_time: pd.Timestamp
    entry: float
    qty: float
    stop: float
    exit_time: pd.Timestamp | None = None
    exit: float | None = None
    reason: str = ""
    pnl: float = 0.0
    extreme: float = 0.0  # highest high (long) / lowest low (short) since entry
    fees: float = 0.0


@dataclass
class Result:
    trades: list = field(default_factory=list)
    equity: pd.Series | None = None


def backtest(d: pd.DataFrame, p: Params, capital: float = 10_000.0) -> Result:
    o, h, l, c = (d[k].to_numpy() for k in ("open", "high", "low", "close"))
    a, ls, ss, tu = d["atr"].to_numpy(), d["long_sig"].to_numpy(), d["short_sig"].to_numpy(), d["trend_up"].to_numpy()
    idx = d.index
    cash = capital
    pos: Trade | None = None
    pending: tuple | None = None   # ("enter", side) or ("exit", reason) to execute at next open
    trades, eq = [], np.empty(len(d))

    def close_pos(i, price, reason):
        nonlocal cash, pos
        fill = price * (1 - p.slippage * pos.side)
        fee = abs(pos.qty) * fill * p.fee
        pos.pnl += pos.side * pos.qty * (fill - pos.entry) - fee
        pos.fees += fee
        cash += pos.side * pos.qty * (fill - pos.entry) - fee
        pos.exit_time, pos.exit, pos.reason = idx[i], fill, reason
        trades.append(pos)
        pos = None

    for i in range(len(d)):
        # 1) execute orders queued at the previous close, at this bar's open
        if pending is not None:
            kind, val = pending
            pending = None
            if kind == "exit" and pos is not None:
                close_pos(i, o[i], val)
            elif kind == "enter" and pos is None:
                side = val
                fill = o[i] * (1 + p.slippage * side)
                stop = fill - side * p.atr_stop * a[i - 1]
                dist = abs(fill - stop)
                if dist > 0 and cash > 0:
                    qty = min(cash * p.risk / dist, cash * p.max_leverage / fill)
                    fee = qty * fill * p.fee
                    cash -= fee
                    pos = Trade(side, idx[i], fill, qty, stop, extreme=fill, fees=fee, pnl=-fee)

        # 2) intrabar stop check (stop hit if price trades through it; gaps fill at open)
        if pos is not None:
            if pos.side == 1 and l[i] <= pos.stop:
                close_pos(i, min(o[i], pos.stop), "stop")
            elif pos.side == -1 and h[i] >= pos.stop:
                close_pos(i, max(o[i], pos.stop), "stop")

        # 3) end-of-bar logic: update trailing stop, check exits/entries for next open
        if pos is not None:
            if pos.side == 1:
                pos.extreme = max(pos.extreme, h[i])
                pos.stop = max(pos.stop, pos.extreme - p.atr_trail * a[i])
                if not tu[i]:
                    pending = ("exit", "trend")
            else:
                pos.extreme = min(pos.extreme, l[i])
                pos.stop = min(pos.stop, pos.extreme + p.atr_trail * a[i])
                if tu[i]:
                    pending = ("exit", "trend")
        elif i + 1 < len(d):
            if ls[i]:
                pending = ("enter", 1)
            elif ss[i]:
                pending = ("enter", -1)

        unreal = pos.side * pos.qty * (c[i] - pos.entry) if pos is not None else 0.0
        eq[i] = cash + unreal

    if pos is not None:
        close_pos(len(d) - 1, c[-1], "end")
        eq[-1] = cash
    return Result(trades, pd.Series(eq, index=idx))


# ------------------------------------------------------------------------ metrics
def stats(res: Result, d: pd.DataFrame, capital: float) -> dict:
    eq = res.equity
    rets = eq.pct_change().dropna()
    years = (eq.index[-1] - eq.index[0]).total_seconds() / (365.25 * 86400)
    total = eq.iloc[-1] / capital - 1
    dd = eq / eq.cummax() - 1
    pnls = np.array([t.pnl for t in res.trades])
    wins, losses = pnls[pnls > 0], pnls[pnls <= 0]
    bh = d["close"].iloc[-1] / d["close"].iloc[0] - 1
    bh_eq = d["close"] / d["close"].iloc[0]
    sharpe = rets.mean() / rets.std() * math.sqrt(BARS_PER_YEAR) if rets.std() > 0 else 0.0
    downside = rets[rets < 0].std()
    sortino = rets.mean() / downside * math.sqrt(BARS_PER_YEAR) if downside and downside > 0 else 0.0
    in_mkt = sum((t.exit_time - t.entry_time).total_seconds() for t in res.trades) / max((eq.index[-1] - eq.index[0]).total_seconds(), 1)
    return {
        "Period": f"{eq.index[0]:%Y-%m-%d} -> {eq.index[-1]:%Y-%m-%d} ({years:.1f} y)",
        "Start equity": f"{capital:,.2f}",
        "End equity": f"{eq.iloc[-1]:,.2f}",
        "Total return": f"{total:+.2%}",
        "CAGR": f"{(eq.iloc[-1] / capital) ** (1 / years) - 1:+.2%}" if years > 0 and eq.iloc[-1] > 0 else "n/a",
        "Max drawdown": f"{dd.min():.2%}",
        "Sharpe (ann.)": f"{sharpe:.2f}",
        "Sortino (ann.)": f"{sortino:.2f}",
        "Trades": len(pnls),
        "Win rate": f"{len(wins) / len(pnls):.1%}" if len(pnls) else "n/a",
        "Avg win / avg loss": f"{wins.mean():,.2f} / {losses.mean():,.2f}" if len(wins) and len(losses) else "n/a",
        "Profit factor": f"{wins.sum() / -losses.sum():.2f}" if len(losses) and losses.sum() < 0 else "inf",
        "Expectancy / trade": f"{pnls.mean():,.2f}" if len(pnls) else "n/a",
        "Fees paid": f"{sum(t.fees for t in res.trades):,.2f}",
        "Time in market": f"{in_mkt:.1%}",
        "Buy & hold return": f"{bh:+.2%}",
        "Buy & hold max DD": f"{(bh_eq / bh_eq.cummax() - 1).min():.2%}",
        "_profitable": total > 0,
        "_beats_bh": total > bh,
    }


def print_stats(title: str, s: dict) -> None:
    print(f"\n=== {title} " + "=" * max(0, 60 - len(title)))
    for k, v in s.items():
        if not k.startswith("_"):
            print(f"  {k:<22} {v}")


def trades_frame(res: Result) -> pd.DataFrame:
    return pd.DataFrame(
        [{"side": "LONG" if t.side == 1 else "SHORT", "entry_time": t.entry_time, "entry": round(t.entry, 4),
          "exit_time": t.exit_time, "exit": round(t.exit, 4), "qty": t.qty, "pnl": round(t.pnl, 2),
          "return_%": round(t.side * (t.exit / t.entry - 1) * 100, 2), "reason": t.reason} for t in res.trades]
    )


def run_segment(df_full: pd.DataFrame, start, end, p: Params, capital: float) -> tuple[Result, dict, pd.DataFrame]:
    """Indicators are computed on the full history (no look-ahead: they only use past bars),
    then the backtest is run on the [start, end) slice so warm-up data is never wasted."""
    d = add_signals(df_full, p)
    d = d[(d.index >= start) & (d.index < end)] if start is not None else d
    d = d.dropna(subset=["atr"])
    res = backtest(d, p, capital)
    return res, stats(res, d, capital), d


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="4H trend-following swing strategy backtester")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--csv", default=None, help="use local OHLC CSV instead of downloading")
    ap.add_argument("--refresh", action="store_true", help="re-download data, ignore cache")
    ap.add_argument("--capital", type=float, default=10_000)
    ap.add_argument("--allow-short", action="store_true")
    ap.add_argument("--oos", type=float, default=0.3, help="fraction of data held out as out-of-sample test")
    ap.add_argument("--plot", action="store_true", help="save equity curve to equity.png (needs matplotlib)")
    for f in ("ema_fast", "ema_slow", "breakout", "adx_min", "atr_len", "atr_stop", "atr_trail", "risk", "max_leverage", "fee", "slippage"):
        ap.add_argument(f"--{f.replace('_', '-')}", type=type(getattr(Params, f)), default=getattr(Params, f))
    args = ap.parse_args()

    p = Params(**{f: getattr(args, f) for f in Params.__dataclass_fields__ if f != "allow_short"},
               allow_short=args.allow_short)
    df = load_data(args)
    name = os.path.basename(args.csv) if args.csv else args.symbol
    print(f"Loaded {len(df)} 4h candles for {name}: {df.index[0]} -> {df.index[-1]}")
    print(f"Params: {p}")

    warm = df.index[min(p.ema_slow, len(df) - 1)]
    split = df.index[int(len(df) * (1 - args.oos))] if args.oos > 0 else None

    res, s, d = run_segment(df, warm, df.index[-1] + pd.Timedelta("1s"), p, args.capital)
    print_stats(f"FULL PERIOD  {name} 4h", s)
    if split is not None:
        _, s_is, _ = run_segment(df, warm, split, p, args.capital)
        _, s_oos, _ = run_segment(df, split, df.index[-1] + pd.Timedelta("1s"), p, args.capital)
        print_stats("IN-SAMPLE", s_is)
        print_stats("OUT-OF-SAMPLE (unseen data)", s_oos)

    tf = trades_frame(res)
    tf.to_csv("trades.csv", index=False)
    res.equity.to_csv("equity.csv", header=["equity"])
    if len(tf):
        print("\nExit reasons:", tf["reason"].value_counts().to_dict())
        print("Last 5 trades:\n" + tf.tail(5).to_string(index=False))

    print("\n=== VERDICT " + "=" * 50)
    verdict = "PROFITABLE" if s["_profitable"] else "NOT PROFITABLE"
    print(f"  Full period: {verdict} ({s['Total return']}, PF {s['Profit factor']}, max DD {s['Max drawdown']})")
    if split is not None:
        print(f"  Out-of-sample: {'PROFITABLE' if s_oos['_profitable'] else 'NOT PROFITABLE'} ({s_oos['Total return']})")
    print(f"  vs buy & hold: {'beats' if s['_beats_bh'] else 'does not beat'} B&H ({s['Buy & hold return']}) "
          f"on return; compare drawdowns ({s['Max drawdown']} vs {s['Buy & hold max DD']}).")
    print("  Saved trades.csv and equity.csv")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
            ax1.plot(res.equity.index, res.equity, label="Strategy")
            ax1.plot(d.index, args.capital * d["close"] / d["close"].iloc[0], label="Buy & hold", alpha=0.6)
            if split is not None:
                ax1.axvline(split, color="gray", ls="--", label="OOS start")
            ax1.set_yscale("log"); ax1.legend(); ax1.set_title(f"{name} 4h trend swing strategy")
            ax2.fill_between(res.equity.index, res.equity / res.equity.cummax() - 1, 0, color="tab:red", alpha=0.4)
            ax2.set_ylabel("Drawdown")
            fig.tight_layout(); fig.savefig("equity.png", dpi=120)
            print("  Saved equity.png")
        except ImportError:
            print("  matplotlib not installed; skipping plot")


if __name__ == "__main__":
    main()
