# 4H Trend-Following Swing Bot + Backtester

`swing_trend_bot.py` is a trend-following swing strategy on 4-hour candles, plus a backtester that tells you whether it is profitable.

## Strategy
- **Trend filter:** EMA50 > EMA200 and close > EMA200 for longs (mirrored for shorts with `--allow-short`)
- **Trend strength:** ADX(14) > 20
- **Entry:** close breaks the 20-candle high (or low), and the order fills at the next candle's open
- **Stops:** initial stop at 2.5×ATR, then a chandelier trailing stop at 3.5×ATR that only moves in your favour
- **Trend exit:** EMA50 crosses back through EMA200
- **Sizing:** each trade risks 1% of equity, with no leverage
- **Costs:** 0.10% fee and 0.05% slippage on every fill

Data comes from Binance's public 4h candles (no API key needed) and is cached in `data/`. You can also pass your own CSV with `--csv`.

## Run
```bash
pip install -r requirements.txt
python swing_trend_bot.py                         # BTCUSDT long-only since 2019
python swing_trend_bot.py --allow-short --plot    # long+short and save equity.png
python swing_trend_bot.py --symbol ETHUSDT --risk 0.02 --atr-trail 3
python swing_trend_bot.py --csv my_ohlc.csv       # columns: time,open,high,low,close[,volume]
```
Each run prints results for the full period, the in-sample part (first 70%) and the out-of-sample part (last 30%), then a verdict. It also writes `trades.csv` and `equity.csv`.

## Backtest results (2019-02 → 2026-11, $10k start, default params)
| Config | Total return | Max DD | Profit factor | Out-of-sample (2024-07 → 2026-11) |
|---|---|---|---|---|
| BTC long-only | +94.5% (8.9% CAGR) | -11.5% | 2.10 | -0.5% |
| BTC long+short | +137.5% | -16.4% | 1.72 | +6.8% |
| ETH long-only | +69.6% | -10.4% | 1.69 | -1.5% |
| ETH long+short | +105.0% | -10.3% | 1.58 | +12.7% |
| BTC buy & hold | +1617% | -77.0% | – | +2.8% |

**What this means:** the strategy made money over the whole period with small drawdowns. Most of that came from 2019–2024, though. On the most recent ~2.4 years, which are held out as unseen data, long-only roughly broke even, and only the long+short versions made money. It never came close to buy & hold's return on BTC, but its drawdowns were far smaller. These are backtests, not guarantees: test it on paper before risking real money.
