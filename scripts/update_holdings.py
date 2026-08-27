import os
import sys
import json
import time
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import finnhub
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent.parent
HOLDINGS_PATH = ROOT / "data" / "holdings.json"
HISTORY_PATH = ROOT / "data" / "portfolio_history.json"
SVG_PATHS = [ROOT / "light_mode.svg", ROOT / "dark_mode.svg"]

NS = "{http://www.w3.org/2000/svg}"

client = None

def api_call(fn, *args, retries=3, **kwargs):
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def get_quote(ticker):
    """Return (current_price, previous_close) or (None, None) on missing data."""
    q = api_call(client.quote, ticker)
    c = q.get("c")
    pc = q.get("pc")
    if not c or not pc:
        return None, None
    return float(c), float(pc)


def get_historical_close(ticker, target, mode):
    """Close price on/before (`last_before`) or on/after (`first_after`) target date.

    Falls back to the closest previous trading day when the exact date was a
    weekend or market holiday, by scanning a window of daily candles.
    """
    start = datetime(target.year, target.month, target.day, tzinfo=timezone.utc) - timedelta(days=14)
    end = datetime(target.year, target.month, target.day, tzinfo=timezone.utc) + timedelta(days=1)
    data = api_call(client.stock_candles, ticker, "D", int(start.timestamp()), int(end.timestamp()))
    if data.get("s") != "ok" or not data.get("t"):
        raise ValueError(f"No candle data for {ticker}")
    ts = data["t"]
    cs = data["c"]
    target_ts = int(datetime(target.year, target.month, target.day, tzinfo=timezone.utc).timestamp())
    if mode == "last_before":
        chosen = None
        for t, c in zip(ts, cs):
            if t <= target_ts:
                chosen = c
            else:
                break
        if chosen is None:
            raise ValueError(f"No trading day on/before {target} for {ticker}")
        return float(chosen)
    else:
        for t, c in zip(ts, cs):
            if t >= target_ts:
                return float(c)
        raise ValueError(f"No trading day on/after {target} for {ticker}")


def load_holdings(path=HOLDINGS_PATH):
    """Load current positions from data/holdings.json.

    The file is manually maintained and contains a flat list of current
    holdings (ticker / shares / average_cost). It is read-only; the script
    never writes to it.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_history(path=HISTORY_PATH):
    """Load portfolio history snapshots from data/portfolio_history.json.

    The file is manually maintained and read-only. The script uses it solely
    to compute the time-weighted return; it never appends or overwrites
    snapshots.
    """
    if not path.exists():
        return {"snapshots": []}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "snapshots" not in data:
        data["snapshots"] = []
    return data


def current_holdings(lots):
    """Map ticker -> current shares from the manually maintained list."""
    holdings = {}
    for lot in lots:
        ticker = lot["ticker"]
        shares = float(lot["shares"])
        if shares > 0:
            holdings[ticker] = shares
    return holdings


def calculate_total_shares(holdings):
    return sum(holdings.values())


def calculate_portfolio_value(holdings, prices):
    return sum(holdings[t] * prices[t] for t in holdings)


def calculate_top_holdings(holdings, prices, count=3):
    """Top holdings ranked by percentage of total portfolio market value.

    Returns the ticker symbols only (the SVG displays the ranking by ticker).
    """
    values = {t: holdings[t] * prices[t] for t in holdings}
    ranked = sorted(values.items(), key=lambda kv: kv[1], reverse=True)
    return [ticker for ticker, _ in ranked[:count]]


def calculate_daily_change(holdings, prices, prev_closes):
    current = sum(holdings[t] * prices[t] for t in holdings)
    previous = sum(holdings[t] * prev_closes[t] for t in holdings)
    return _pct_change(current, previous)


def calculate_month_change(holdings, prices, month_hist):
    current = sum(holdings[t] * prices[t] for t in holdings)
    previous = sum(holdings[t] * month_hist[t] for t in holdings)
    return _pct_change(current, previous)


def calculate_ytd_change(holdings, prices, ytd_hist):
    current = sum(holdings[t] * prices[t] for t in holdings)
    previous = sum(holdings[t] * ytd_hist[t] for t in holdings)
    return _pct_change(current, previous)


def calculate_twr(snapshots):
    """Time-weighted return since the first recorded snapshot.

    Computed exclusively from data/portfolio_history.json (read-only). External
    cash flows (deposits/withdrawals) distort a naive "current vs first value"
    comparison, so we compound the return of each sub-period instead.

    For each sub-period the external cash flow is assumed to enter at the
    BEGINNING of the period:

        period_net_flow = current_deposits - previous_deposits
        capital_base     = previous_portfolio_value + period_net_flow
        period_return    = (current_portfolio_value - capital_base) / capital_base

    The first snapshot is treated as inception (previous value and previous
    deposits are both 0), so its capital base is its own cumulative deposits.

    Period returns are chained by multiplying (1 + return) across all periods
    and subtracting 1 at the end.
    """
    seq = sorted(snapshots, key=lambda s: s["date"])
    if not seq:
        return 0.0

    growth = 1.0
    prev_value = 0.0
    prev_deposits = 0.0
    for snap in seq:
        value = float(snap["portfolio_value"])
        deposits = float(snap.get("deposits", prev_deposits))
        period_net_flow = deposits - prev_deposits
        capital_base = prev_value + period_net_flow
        if capital_base:
            period_return = (value - capital_base) / capital_base
        else:
            period_return = 0.0
        growth *= (1 + period_return)
        prev_value = value
        prev_deposits = deposits
    return (growth - 1) * 100


def _pct_change(current, previous):
    if not previous:
        return 0.0
    return (current / previous - 1) * 100


def history_value_near(snapshots, target):
    """Portfolio value of the snapshot closest in date to `target`.

    Used as a fallback when per-ticker historical candles are unavailable
    (e.g. Finnhub free tier blocks /stock/candles). Picks the snapshot with the
    smallest absolute day difference to `target`.
    """
    seq = sorted(snapshots, key=lambda s: s["date"])
    if not seq:
        return None
    best = None
    best_diff = None
    for s in seq:
        try:
            d = datetime.strptime(s["date"], "%Y-%m-%d").date()
        except ValueError:
            continue
        diff = abs((d - target).days)
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best = float(s["portfolio_value"])
    return best


def history_pct_change(snapshots, current_value, target):
    past = history_value_near(snapshots, target)
    if past is None or not past:
        return 0.0
    return (current_value / past - 1) * 100


def format_pct(value):
    if value > 0:
        return f"+{value:.2f}%"
    if value < 0:
        return f"{value:.2f}%"
    return "0.00%"


def format_shares(value):
    text = f"{value:.4f}"
    text = text.rstrip("0").rstrip(".")
    return text


def determine_market_date():
    """Latest US trading day. Returns None on weekends / market holidays."""
    eastern = datetime.now(ZoneInfo("America/New_York"))
    d = eastern.date()
    if d.weekday() >= 5:
        return None
    try:
        holidays = api_call(client.market_holiday, "US")
        holiday_dates = {h["date"] for h in holidays}
        if d.isoformat() in holiday_dates:
            return None
    except Exception:
        pass
    return d

LABELS = {
    "share-number": "Shares.Number:",
    "share-value": "Value:",
    "share-today": "Today:",
    "share-month": "1M:",
    "share-ytd": "YTD:",
    "share-all": "ALL:",
    "date": "Last.Update:",
}
TOP_LABELS = ["Shares.Top[0]:", "Shares.Top[1]:", "Shares.Top[2]:"]
ALIGN_COL = 36

def update_svg(svg_path, metrics):
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    tree = ET.parse(svg_path)
    root = tree.getroot()

    by_id = {}
    for el in root.iter():
        eid = el.get("id")
        if eid:
            by_id.setdefault(eid, []).append(el)

    text_tag = NS + "text"

    def find_filler(el):
        """Return the filler `<tspan class="cc">` immediately preceding `el`."""
        for text_el in root.iter(text_tag):
            children = list(text_el)
            for i, c in enumerate(children):
                if c is el:
                    for j in range(i - 1, -1, -1):
                        if (children[j].get("class") or "") == "cc":
                            return children[j]
        return None

    def align_filler(el, label, text):
        filler = find_filler(el)
        if filler is None:
            return
        dots = ALIGN_COL - len(label) - len(text) - 2
        if dots < 0:
            dots = 0
        filler.text = " " + "." * dots + " "

    def set_metric(eid, text, colored=False):
        els = by_id.get(eid)
        if not els or len(els) != 1:
            raise RuntimeError(f"SVG {svg_path.name}: expected exactly one id '{eid}', found {len(els) if els else 0}")
        el = els[0]
        if colored:
            el.set("class", "delColor" if text.startswith("-") else "addColor")
        el.text = text
        label = LABELS.get(eid)
        if label is not None:
            align_filler(el, label, text)

    set_metric("date", metrics["date"])
    set_metric("share-number", metrics["total_shares"])
    set_metric("share-value", metrics["value"])
    set_metric("share-today", metrics["today"], colored=True)
    set_metric("share-month", metrics["month"], colored=True)
    set_metric("share-ytd", metrics["ytd"], colored=True)
    set_metric("share-all", metrics["all_time"], colored=True)

    tops = by_id.get("share-top")
    if not tops or len(tops) != 3:
        raise RuntimeError(f"SVG {svg_path.name}: expected 3 'share-top' elements, found {len(tops) if tops else 0}")
    for el, label, txt in zip(tops, TOP_LABELS, metrics["top3"]):
        el.text = txt
        align_filler(el, label, txt)

    tree.write(svg_path, encoding="utf-8", xml_declaration=True)


def load_env_file(path=ROOT / ".env"):
    """Best-effort load of a local .env file (quotes stripped).

    Only sets variables that are not already present in the environment, so an
    explicit FINNHUB_API_KEY (e.g. the GitHub Actions secret) always wins. The
    API key is never printed.
    """
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


def main():
    global client
    load_env_file()
    api_key = os.environ.get("FINNHUB_API_KEY")
    if not api_key:
        sys.exit("ERROR: FINNHUB_API_KEY environment variable is not set")
    client = finnhub.Client(api_key=api_key)

    lots = load_holdings()
    holdings = current_holdings(lots)
    if not holdings:
        sys.exit("ERROR: no current holdings found in configuration")

    market_date = determine_market_date()
    if market_date is None:
        print("No new US trading session. Nothing to update.")
        return

    prices = {}
    prev_closes = {}
    for ticker in holdings:
        cur, pc = get_quote(ticker)
        if cur is None or pc is None:
            sys.exit(f"ERROR: Unable to retrieve market data for {ticker}")
        prices[ticker] = cur
        prev_closes[ticker] = pc

    month_target = market_date - timedelta(days=30)
    ytd_target = date(market_date.year, 1, 1)

    history = load_history()

    month_hist = {}
    ytd_hist = {}
    use_candles = True
    for ticker in holdings:
        try:
            month_hist[ticker] = get_historical_close(ticker, month_target, "last_before")
            ytd_hist[ticker] = get_historical_close(ticker, ytd_target, "first_after")
        except Exception:
            use_candles = False
            break

    current_value = calculate_portfolio_value(holdings, prices)
    today = calculate_daily_change(holdings, prices, prev_closes)
    if use_candles:
        month = calculate_month_change(holdings, prices, month_hist)
        ytd = calculate_ytd_change(holdings, prices, ytd_hist)
    else:
        print("NOTE: historical candles unavailable (Finnhub free tier); "
              "deriving 1M/YTD from data/portfolio_history.json snapshots.")
        month = history_pct_change(history["snapshots"], current_value, month_target)
        ytd = history_pct_change(history["snapshots"], current_value, ytd_target)

    top3 = calculate_top_holdings(holdings, prices, count=3)
    total_shares = calculate_total_shares(holdings)

    all_time = calculate_twr(history["snapshots"])

    metrics = {
        "date": market_date.isoformat(),
        "total_shares": str(int(round(total_shares))),
        "value": f"$ {current_value:,.2f}",
        "today": format_pct(today),
        "month": format_pct(month),
        "ytd": format_pct(ytd),
        "all_time": format_pct(all_time),
        "top3": top3,
    }

    for svg_path in SVG_PATHS:
        update_svg(svg_path, metrics)

    print(f"Updated portfolio for {market_date.isoformat()} (value ${current_value:,.2f})")


if __name__ == "__main__":
    main()
