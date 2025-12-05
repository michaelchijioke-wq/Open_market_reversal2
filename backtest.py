"""
Simple backtest runner for the sandbox strategy.

- Loads US500 data from the local CSV log (ticks_us500.csv) when available.
- Falls back to requesting a small MT5 history range when the CSV is missing.
- Simulates trade entries on BUY/SELL signals from strategy.generate_signal().
- Uses a fixed position size plus fixed stop loss and take profit in points.
- Prints summary stats (PnL, drawdown, trade count, win rate) after the run.

Run with: python backtest.py
"""

import csv
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import MetaTrader5 as mt5

from daily_context import DailyContext, build_daily_context
from opening_range import compute_opening_range
from box_zones import classify_box_zone, compute_box_position
from strategy import generate_signal

# Files and defaults
CSV_FILE = "ticks_us500.csv"
SYMBOL = "US500"

# Backtest parameters
POSITION_SIZE = 1.0  # Contracts/units per trade
STOP_LOSS_POINTS = 20.0  # Stop distance in price points
TAKE_PROFIT_POINTS = 40.0  # Take-profit distance in price points

# How far back to fetch MT5 bars when CSV data is missing (in days)
MT5_LOOKBACK_DAYS = 2


@dataclass
class Position:
    side: str  # "LONG" or "SHORT"
    entry_price: float
    stop_price: float
    take_profit: float
    entry_time: str


@dataclass
class Trade:
    side: str
    entry_price: float
    exit_price: float
    profit: float
    entry_time: str
    exit_time: str


def connect_mt5() -> bool:
    """Initialize MT5 and return True on success."""

    if mt5.initialize():
        return True
    print(f"[WARN] Could not initialize MT5: {mt5.last_error()}")
    return False


def shutdown_mt5() -> None:
    """Shutdown MT5 quietly."""

    mt5.shutdown()


def load_ticks_from_csv(path: str) -> List[SimpleNamespace]:
    """Load ticks from CSV. Returns an empty list on failure."""

    if not os.path.exists(path):
        print(f"[INFO] CSV file not found: {path}")
        return []

    ticks: List[SimpleNamespace] = []
    try:
        with open(path, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                try:
                    ticks.append(
                        SimpleNamespace(
                            time=row.get("utc_time", ""),
                            bid=float(row.get("bid", 0) or 0),
                            ask=float(row.get("ask", 0) or 0),
                            last=float(row.get("last", 0) or 0),
                            volume=float(row.get("volume", 0) or 0),
                        )
                    )
                except ValueError:
                    # Skip rows with bad numeric fields
                    continue
    except OSError as exc:
        print(f"[WARN] Could not read CSV data ({exc}).")
        return []

    print(f"[INFO] Loaded {len(ticks)} ticks from CSV")
    return ticks


def load_ticks_from_mt5(symbol: str) -> List[SimpleNamespace]:
    """Load simple tick-like data from recent MT5 rates when CSV is absent."""

    if not connect_mt5():
        return []

    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=MT5_LOOKBACK_DAYS)

    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, start, end)
    if rates is None:
        print(f"[WARN] MT5 returned no data for {symbol}: {mt5.last_error()}")
        shutdown_mt5()
        return []

    ticks = [
        SimpleNamespace(
            time=datetime.fromtimestamp(rate["time"], tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            bid=float(rate["close"]),
            ask=float(rate["close"]),
            last=float(rate["close"]),
            volume=float(rate["real_volume"] if "real_volume" in rate.dtype.names else rate["tick_volume"]),
        )
        for rate in rates
    ]

    shutdown_mt5()
    print(f"[INFO] Loaded {len(ticks)} derived ticks from MT5 rates")
    return ticks


def pick_ticks() -> Tuple[List[SimpleNamespace], str]:
    """Return tick data and a description of the source used."""

    csv_ticks = load_ticks_from_csv(CSV_FILE)
    if csv_ticks:
        return csv_ticks, "CSV"

    mt5_ticks = load_ticks_from_mt5(SYMBOL)
    if mt5_ticks:
        return mt5_ticks, "MT5"

    return [], "NONE"


def build_daily_bars_from_ticks(ticks: List[SimpleNamespace]) -> List[SimpleNamespace]:
    """Aggregate ticks into simple daily bars for context building.

    Uses mid-price (average of bid/ask) and sums tick volumes for the day.
    """

    daily: Dict[str, Dict[str, float]] = {}
    for tick in ticks:
        tick_dt = parse_time(getattr(tick, "time", ""))
        if not tick_dt:
            continue

        day_key = tick_dt.strftime("%Y-%m-%d")
        mid_price = (tick.bid + tick.ask) / 2.0
        record = daily.setdefault(
            day_key,
            {"high": mid_price, "low": mid_price, "close": mid_price, "volume": 0.0},
        )

        record["high"] = max(record["high"], mid_price)
        record["low"] = min(record["low"], mid_price)
        record["close"] = mid_price
        record["volume"] += float(getattr(tick, "volume", 0.0) or 0.0)

    daily_bars: List[SimpleNamespace] = []
    for day_str in sorted(daily.keys()):
        dt = datetime.strptime(day_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        fields = daily[day_str]
        daily_bars.append(
            SimpleNamespace(
                time=dt.timestamp(),
                high=fields["high"],
                low=fields["low"],
                close=fields["close"],
                tick_volume=fields["volume"],
            )
        )

    return daily_bars


def build_daily_atr_map(daily_bars: List[SimpleNamespace]) -> Dict[str, float]:
    """Return a map of YYYY-MM-DD -> ATR14 using prior daily ranges."""

    atr_map: Dict[str, float] = {}
    ranges: List[float] = []

    for idx, bar in enumerate(daily_bars):
        prior_ranges = ranges[max(0, idx - 14) : idx]
        atr = sum(prior_ranges[-14:]) / 14.0 if len(prior_ranges) >= 14 else 0.0

        day_key = datetime.fromtimestamp(bar.time, tz=timezone.utc).strftime("%Y-%m-%d")
        atr_map[day_key] = atr
        ranges.append(bar.high - bar.low)

    return atr_map


def build_mock_daily_context(
    ticks: List[SimpleNamespace], day_key: str
) -> Optional[Tuple[DailyContext, float]]:
    """
    Create a simple mock DailyContext for single-day CSV tests.

    When the CSV only contains one trading session, we have no real previous-day
    candle. To exercise the full pipeline anyway, we approximate "yesterday" by
    using the first 60 minutes of prices in the file. The range of that window is
    used as both the previous high/low and as a rough ATR14 placeholder.
    """

    if not ticks:
        return None

    # Collect and sort ticks by time so the 60-minute slice is deterministic.
    dated_ticks: List[Tuple[datetime, SimpleNamespace]] = []
    for tick in ticks:
        tick_dt = parse_time(getattr(tick, "time", ""))
        if tick_dt:
            dated_ticks.append((tick_dt, tick))

    if not dated_ticks:
        return None

    dated_ticks.sort(key=lambda item: item[0])
    first_dt = dated_ticks[0][0]
    window_end = first_dt + timedelta(minutes=60)

    prices: List[float] = []
    for tick_dt, tick in dated_ticks:
        if tick_dt > window_end:
            break

        bid = getattr(tick, "bid", 0.0)
        ask = getattr(tick, "ask", 0.0)
        prices.append((bid + ask) / 2.0)

    if not prices:
        return None

    prev_low_mock = min(prices)
    prev_high_mock = max(prices)
    daily_atr_mock = prev_high_mock - prev_low_mock

    context = DailyContext(
        date=day_key,
        prev_high=prev_high_mock,
        prev_low=prev_low_mock,
        daily_atr_14=daily_atr_mock,
        is_trading_day=True,
    )

    print(
        "[CONTEXT-MOCK] Using single-day mock context: "
        f"prev_high={prev_high_mock:.2f} | prev_low={prev_low_mock:.2f} | "
        f"ATR(14)={daily_atr_mock:.2f}"
    )

    # Return both the context and the mock ATR so downstream logs stay consistent.
    return context, daily_atr_mock


def build_daily_context_map(
    daily_bars: List[SimpleNamespace],
    atr_map: Dict[str, float],
) -> Dict[str, DailyContext]:
    """Return a map of YYYY-MM-DD -> DailyContext using the prior day's bar."""

    contexts: Dict[str, DailyContext] = {}

    for idx in range(1, len(daily_bars)):
        today_bar = daily_bars[idx]
        yesterday = daily_bars[idx - 1]

        day_key = datetime.fromtimestamp(today_bar.time, tz=timezone.utc).strftime("%Y-%m-%d")
        is_trading_day = (yesterday.high > yesterday.low) and (
            getattr(yesterday, "tick_volume", 0) > 0
        )

        contexts[day_key] = DailyContext(
            date=day_key,
            prev_high=yesterday.high,
            prev_low=yesterday.low,
            daily_atr_14=atr_map.get(day_key, 0.0),
            is_trading_day=is_trading_day,
        )

    return contexts


def update_candle_window(
    tick_dt: Optional[datetime],
    mid_price: float,
    volume: float,
    current_bar: Optional[SimpleNamespace],
    recent_bars: List[SimpleNamespace],
    max_bars: int = 5,
) -> Tuple[Optional[SimpleNamespace], List[SimpleNamespace]]:
    """Maintain a small rolling window of simple 1-minute candles.

    The window helps the strategy run pattern detection without external feeds.
    """

    if tick_dt is None:
        return current_bar, recent_bars

    minute_key = tick_dt.replace(second=0, microsecond=0)
    mid_volume = float(volume or 0.0)

    if current_bar is None or getattr(current_bar, "time", None) != minute_key:
        # Finalize the previous bar into the window.
        if current_bar is not None:
            recent_bars.append(current_bar)
            recent_bars = recent_bars[-max_bars:]

        current_bar = SimpleNamespace(
            time=minute_key.strftime("%Y-%m-%d %H:%M:%S"),
            open=mid_price,
            high=mid_price,
            low=mid_price,
            close=mid_price,
            volume=mid_volume,
        )
    else:
        current_bar.high = max(current_bar.high, mid_price)
        current_bar.low = min(current_bar.low, mid_price)
        current_bar.close = mid_price
        current_bar.volume += mid_volume

    return current_bar, recent_bars


def maybe_close(position: Position, bid: float, ask: float, tick_time: str) -> Tuple[Optional[Trade], Optional[Position]]:
    """Close a position if stop loss or take profit is hit at this tick."""

    if position is None:
        return None, None

    exit_price: Optional[float] = None

    if position.side == "LONG":
        if bid <= position.stop_price:
            exit_price = bid
        elif bid >= position.take_profit:
            exit_price = bid
    elif position.side == "SHORT":
        if ask >= position.stop_price:
            exit_price = ask
        elif ask <= position.take_profit:
            exit_price = ask

    if exit_price is None:
        return None, position

    profit = (exit_price - position.entry_price) * POSITION_SIZE if position.side == "LONG" else (position.entry_price - exit_price) * POSITION_SIZE

    trade = Trade(
        side=position.side,
        entry_price=position.entry_price,
        exit_price=exit_price,
        profit=profit,
        entry_time=position.entry_time,
        exit_time=tick_time,
    )
    return trade, None


def open_position(signal: str, bid: float, ask: float, tick_time: str) -> Optional[Position]:
    """Open a new position based on the signal."""

    if signal == "BUY":
        entry = ask
        return Position(
            side="LONG",
            entry_price=entry,
            stop_price=entry - STOP_LOSS_POINTS,
            take_profit=entry + TAKE_PROFIT_POINTS,
            entry_time=tick_time,
        )

    if signal == "SELL":
        entry = bid
        return Position(
            side="SHORT",
            entry_price=entry,
            stop_price=entry + STOP_LOSS_POINTS,
            take_profit=entry - STOP_LOSS_POINTS,
            entry_time=tick_time,
        )

    return None


def update_stats(equity: float, profit: float, peak: float, max_dd: float) -> Tuple[float, float, float]:
    """Return updated equity, peak equity, and drawdown after a trade."""

    new_equity = equity + profit
    peak = max(peak, new_equity)
    drawdown = max(max_dd, peak - new_equity)
    return new_equity, peak, drawdown


def run_backtest(
    ticks: List[SimpleNamespace],
    daily_context_map: Dict[str, DailyContext],
) -> None:
    """Process ticks, generate signals, and simulate trades."""

    context: Dict = {}
    position: Optional[Position] = None
    trades: List[Trade] = []
    equity = 0.0
    peak_equity = 0.0
    max_drawdown = 0.0
    equity_history: List[Tuple[int, float]] = []
    recent_candles: List[SimpleNamespace] = []
    current_bar: Optional[SimpleNamespace] = None

    for tick in ticks:
        bid = tick.bid
        ask = tick.ask
        tick_time = tick.time
        tick_dt = parse_time(tick_time)
        mid_price = (bid + ask) / 2.0

        # Maintain a small rolling set of 1-minute candles for pattern checks.
        current_bar, recent_candles = update_candle_window(
            tick_dt, mid_price, getattr(tick, "volume", 0.0), current_bar, recent_candles
        )
        candles_for_signal = recent_candles + ([current_bar] if current_bar else [])

        # Step 1: evaluate exits on the current position.
        trade_closed, position = maybe_close(position, bid, ask, tick_time)
        if trade_closed:
            trades.append(trade_closed)
            equity, peak_equity, max_drawdown = update_stats(
                equity, trade_closed.profit, peak_equity, max_drawdown
            )
            equity_history.append((len(trades), equity))

        # Step 2: compute box position for context and run the strategy with patterns.
        day_key = tick_dt.strftime("%Y-%m-%d") if tick_dt else ""
        day_ctx = daily_context_map.get(day_key)
        zone = None
        if day_ctx is None:
            print(f"[BOX] {tick_time} | No prior day context; treating zone as unknown")
        else:
            pos = compute_box_position(mid_price, day_ctx.prev_low, day_ctx.prev_high)
            zone = classify_box_zone(pos)
            print(
                f"[BOX] Price: {mid_price:.2f} | Pos: {pos:.2f} | Zone: {zone} "
                f"| Prev High/Low: {day_ctx.prev_high:.2f}/{day_ctx.prev_low:.2f}"
            )

        signal, details = generate_signal(
            tick,
            context,
            zone=zone,
            recent_candles=candles_for_signal,
            daily_context=day_ctx,
        )

        base_signal = details.get("base_signal", "FLAT") if isinstance(details, dict) else "FLAT"
        print(
            f"[BASE] {tick_time} | base={base_signal} | prev_close={details.get('prev_close', 0) or 0:.2f} "
            f"| curr_close={details.get('curr_close', 0) or 0:.2f}"
        )

        patterns = details.get("patterns", []) if isinstance(details, dict) else []
        reason = details.get("reason", "") if isinstance(details, dict) else ""

        if patterns and signal in ("BUY", "SELL"):
            primary = patterns[0]
            name = getattr(primary, "name", None) or primary.get("name")
            direction = getattr(primary, "direction", None) or primary.get("direction")
            confidence = getattr(primary, "confidence", None) or primary.get("confidence", 0.0)
            print(
                f"[PATTERN] {tick_time} | {name} | direction={direction.upper()} | confidence={confidence:.2f}"
            )
        elif reason:
            print(f"[PATTERN] {tick_time} | {reason} -> SKIP")

        pattern_state = "OK" if signal in ("BUY", "SELL") and patterns else "NONE"
        decision = "TAKEN" if signal in ("BUY", "SELL") else "SKIP"
        print(
            f"[DECISION] base={base_signal} zone={zone or 'UNKNOWN'} pattern={pattern_state} "
            f"reason={reason or 'n/a'} -> {decision}"
        )

        effective_signal = signal

        if position and effective_signal in ("BUY", "SELL") and ((position.side == "LONG" and effective_signal == "SELL") or (position.side == "SHORT" and effective_signal == "BUY")):
            # Close existing position before switching direction.
            exit_price = bid if position.side == "LONG" else ask
            profit = (exit_price - position.entry_price) * POSITION_SIZE if position.side == "LONG" else (position.entry_price - exit_price) * POSITION_SIZE
            trades.append(
                Trade(
                    side=position.side,
                    entry_price=position.entry_price,
                    exit_price=exit_price,
                    profit=profit,
                    entry_time=position.entry_time,
                    exit_time=tick_time,
                )
            )
            equity, peak_equity, max_drawdown = update_stats(equity, profit, peak_equity, max_drawdown)
            equity_history.append((len(trades), equity))
            position = None

        if position is None:
            position = open_position(effective_signal, bid, ask, tick_time)

    # Close any open position at the last available price.
    if position and ticks:
        last_tick = ticks[-1]
        exit_price = last_tick.bid if position.side == "LONG" else last_tick.ask
        profit = (exit_price - position.entry_price) * POSITION_SIZE if position.side == "LONG" else (position.entry_price - exit_price) * POSITION_SIZE
        trades.append(
            Trade(
                side=position.side,
                entry_price=position.entry_price,
                exit_price=exit_price,
                profit=profit,
                entry_time=position.entry_time,
                exit_time=last_tick.time,
            )
        )
        equity, peak_equity, max_drawdown = update_stats(equity, profit, peak_equity, max_drawdown)
        equity_history.append((len(trades), equity))

    print_results(trades, equity, max_drawdown, equity_history)


def parse_time(value: str) -> Optional[datetime]:
    """Try to parse time strings in the expected format.

    Handles both "YYYY-MM-DD HH:MM:SS" and the same with a trailing " UTC".
    """

    if not value:
        return None

    text = str(value).strip()

    # First, try plain "YYYY-MM-DD HH:MM:SS"
    try:
        return datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        pass

    # If it ends with " UTC", strip that and try again.
    if text.endswith(" UTC"):
        try:
            return datetime.strptime(text[:-4], "%Y-%m-%d %H:%M:%S")
        except (TypeError, ValueError):
            return None

    return None


def group_ticks_by_day(ticks: List[SimpleNamespace]) -> Dict[str, List[SimpleNamespace]]:
    """Return ticks bucketed by YYYY-MM-DD string."""

    grouped: Dict[str, List[SimpleNamespace]] = {}
    for tick in ticks:
        tick_dt = parse_time(getattr(tick, "time", ""))
        if not tick_dt:
            continue

        day_key = tick_dt.strftime("%Y-%m-%d")
        grouped.setdefault(day_key, []).append(tick)

    return grouped


def log_opening_ranges(
    grouped_ticks: Dict[str, List[SimpleNamespace]], daily_atr_map: Dict[str, float]
) -> None:
    """Compute and print opening range diagnostics for each day."""

    for day in sorted(grouped_ticks.keys()):
        atr = daily_atr_map.get(day, 0.0)
        result = compute_opening_range(grouped_ticks[day], atr)

        if result is None:
            print(f"[OPENING_RANGE] Date {day} | No ticks in window")
            continue

        print(
            "[OPENING_RANGE] "
            f"Date {result.date} | OR: {result.opening_range_points:.2f} | "
            f"ATR14: {atr:.2f} | OR%ATR: {result.opening_range_pct_atr:.2f} | "
            f"OK: {result.opening_range_ok}"
        )


def build_opening_range_map(
    grouped_ticks: Dict[str, List[SimpleNamespace]],
    daily_atr_map: Dict[str, float],
) -> Dict[str, object]:
    """Return a map of YYYY-MM-DD -> opening range results for each trading day.

    Skips days where the opening window has no ticks or ATR is zero.
    """

    opening_range_map: Dict[str, object] = {}

    for day in sorted(grouped_ticks.keys()):
        atr = daily_atr_map.get(day, 0.0)
        result = compute_opening_range(grouped_ticks[day], atr)
        if result is None:
            continue

        opening_range_map[day] = result

    return opening_range_map


def ensure_context_with_mock(
    grouped_ticks: Dict[str, List[SimpleNamespace]],
    daily_context_map: Dict[str, DailyContext],
    daily_atr_map: Dict[str, float],
) -> None:
    """
    Add a mock DailyContext for days without a prior day.

    This is only for single-day CSV backtests so we can exercise the full box and
    pattern pipeline. It approximates "yesterday" using the first 60 minutes of
    prices and writes the mock into both the context map and ATR map.
    """

    for day_key, ticks in grouped_ticks.items():
        if day_key in daily_context_map:
            continue

        mock = build_mock_daily_context(ticks, day_key)
        if mock is None:
            continue

        context, mock_atr = mock
        daily_context_map[day_key] = context
        daily_atr_map[day_key] = mock_atr


def print_results(
    trades: List[Trade], total_pnl: float, max_drawdown: float, equity_history: List[Tuple[int, float]]
) -> None:
    """Display summary statistics and simple diagnostics."""

    trade_count = len(trades)
    wins = sum(1 for t in trades if t.profit > 0)
    win_rate = (wins / trade_count * 100) if trade_count else 0.0
    avg_pnl = (total_pnl / trade_count) if trade_count else 0.0
    long_trades = sum(1 for t in trades if t.side == "LONG")
    short_trades = sum(1 for t in trades if t.side == "SHORT")

    durations = []
    for t in trades:
        start = parse_time(t.entry_time)
        end = parse_time(t.exit_time)
        if start and end:
            durations.append((end - start).total_seconds())
    avg_duration = sum(durations) / len(durations) if durations else 0.0

    print("\n=== Backtest Summary ===")
    print(f"Trades: {trade_count}")
    print(f"Win rate: {win_rate:.1f}%")
    print(f"Total PnL: {total_pnl:.2f}")
    print(f"Average PnL per trade: {avg_pnl:.2f}")
    print(f"Max drawdown: {max_drawdown:.2f}")
    print(f"LONG trades: {long_trades} | SHORT trades: {short_trades}")
    print(f"Average trade duration (sec): {avg_duration:.1f}")

    if trade_count:
        print("\nPnL for last 10 trades:")
        for idx, trade in enumerate(trades[-10:], start=max(trade_count - 9, 1)):
            print(f"#{idx}: {trade.profit:.2f} ({trade.side})")

        print("\nEquity curve (every 10 trades):")
        for trade_idx, equity in equity_history:
            if trade_idx % 10 == 0:
                print(f"After trade {trade_idx}: Equity {equity:.2f}")

        print("\nSample trades (first 5):")
        for trade in trades[:5]:
            print(
                f"{trade.side} | Entry {trade.entry_price:.2f} @ {trade.entry_time} | "
                f"Exit {trade.exit_price:.2f} @ {trade.exit_time} | PnL {trade.profit:.2f}"
            )


def summarize_daily_context(daily_bars: List[SimpleNamespace]) -> Optional[DailyContext]:
    """Build and print a simple daily context snapshot from aggregated bars."""

    context = build_daily_context(daily_bars)
    if context is None:
        print("[INFO] No daily context available (not enough data).")
        return None

    print(
        f"[INFO] Daily context -> Date: {context.date} | Prev High: {context.prev_high:.2f} | "
        f"Prev Low: {context.prev_low:.2f} | ATR14: {context.daily_atr_14:.2f} | "
        f"Trading day: {context.is_trading_day}"
    )
    return context


def main() -> None:
    ticks, source = pick_ticks()
    if not ticks:
        print("[ERROR] No data available from CSV or MT5. Backtest aborted.")
        return

    print(f"[INFO] Running backtest on {len(ticks)} ticks (source: {source})")

    # Build a simple daily context snapshot from available ticks.
    daily_bars = build_daily_bars_from_ticks(ticks)
    summarize_daily_context(daily_bars)

    # Group ticks by day so we can derive mock context when no prior day exists.
    grouped_ticks = group_ticks_by_day(ticks)

    # Build ATRs and real context maps from daily bars.
    daily_atr_map = build_daily_atr_map(daily_bars)
    daily_context_map = build_daily_context_map(daily_bars, daily_atr_map)

    # Patch missing days (like our single-day CSV) with a mock context so that
    # box zones and pattern logic can still run.
    ensure_context_with_mock(grouped_ticks, daily_context_map, daily_atr_map)

    # (Optional) opening-range diagnostics – leave as-is if this already exists.
    # log_opening_ranges(grouped_ticks, daily_atr_map)

    run_backtest(ticks, daily_context_map)


if __name__ == "__main__":
    main()
