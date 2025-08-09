# analyze_trades.py
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd


def load_trades(path: Path) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # dict -> list of rows
    rows: List[Dict[str, Any]] = []
    for _, t in data.items():
        row = {**t}
        # Flatten classifier probs if present
        if isinstance(t.get("classifier_conf"), dict):
            for k, v in t["classifier_conf"].items():
                row[k] = v
        # meta_classifier: assumed order [Reject, 1:1, 1:2, 1:3]
        if isinstance(t.get("meta_classifier"), list):
            mc = t["meta_classifier"]
            # guard length
            for i, name in enumerate(["meta_reject_prob", "meta_1_1_prob", "meta_1_2_prob", "meta_1_3_prob"][: len(mc)]):
                row[name] = mc[i]
            row["meta_max_prob"] = max(mc) if mc else np.nan
        rows.append(row)

    df = pd.DataFrame(rows)

    # Parse timestamps, safe coercion
    for col in ["entry_time", "exit_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    # Fallbacks for equity/balance at entry
    if "equity_at_entry" not in df.columns:
        df["equity_at_entry"] = np.nan
    if "balance_at_entry" not in df.columns:
        df["balance_at_entry"] = np.nan

    # Choose equity_at_entry if available, else balance_at_entry
    df["equity_base"] = df["equity_at_entry"].fillna(df["balance_at_entry"])

    # Clean trade_type
    df["trade_type"] = df["trade_type"].str.lower()

    # Profit must be numeric (assumed realized)
    df["profit"] = pd.to_numeric(df["profit"], errors="coerce").fillna(0.0)
    # Duration (minutes) – recompute from timestamps if available
    if "entry_time" in df.columns and "exit_time" in df.columns:
        dt = (df["exit_time"] - df["entry_time"]).dt.total_seconds() / 60.0
        df["trade_duration_min"] = df.get("trade_duration", dt).fillna(dt)
    else:
        df["trade_duration_min"] = pd.to_numeric(df.get("trade_duration"), errors="coerce")

    # Prices numeric
    for col in ["entry_price", "exit_price", "sl"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df
def class_impact_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each meta_class in df:
    - Show its own PF, win rate, EV
    - Show overall PF/EV if that class were removed
    """
    results = []
    classes = sorted(df["meta_class"].dropna().unique())
    for c in classes:
        sub = df[df["meta_class"] == c]
        excl = df[df["meta_class"] != c]

        # PF for this class
        gp = sub.loc[sub["profit"] > 0, "profit"].sum()
        gl = sub.loc[sub["profit"] < 0, "profit"].abs().sum()
        pf_class = np.inf if gl == 0 and gp > 0 else (gp / gl if gl > 0 else np.nan)

        # EV for this class
        p_win = sub["is_win"].mean()
        avg_win = sub.loc[sub["is_win"], "profit"].mean()
        avg_loss = sub.loc[sub["is_loss"], "profit"].mean()
        ev_class = (p_win * (avg_win if not np.isnan(avg_win) else 0)) + ((1 - p_win) * (avg_loss if not np.isnan(avg_loss) else 0))

        # PF without this class
        gp_excl = excl.loc[excl["profit"] > 0, "profit"].sum()
        gl_excl = excl.loc[excl["profit"] < 0, "profit"].abs().sum()
        pf_excl = np.inf if gl_excl == 0 and gp_excl > 0 else (gp_excl / gl_excl if gl_excl > 0 else np.nan)

        # EV without this class
        p_win_excl = excl["is_win"].mean()
        avg_win_excl = excl.loc[excl["is_win"], "profit"].mean()
        avg_loss_excl = excl.loc[excl["is_loss"], "profit"].mean()
        ev_excl = (p_win_excl * (avg_win_excl if not np.isnan(avg_win_excl) else 0)) + ((1 - p_win_excl) * (avg_loss_excl if not np.isnan(avg_loss_excl) else 0))

        results.append({
            "meta_class": int(c),
            "class_pf": pf_class,
            "class_ev": ev_class,
            "class_win_rate": p_win,
            "overall_pf_without_class": pf_excl,
            "overall_ev_without_class": ev_excl,
            "trades_in_class": len(sub)
        })
    return pd.DataFrame(results)


def filter_trades(df: pd.DataFrame, threshold: float, include_reject: bool, allowed_classes=None) -> pd.DataFrame:
    # Threshold filter
    if "meta_max_prob" in df.columns:
        filt = df["meta_max_prob"] >= threshold
    else:
        filt = pd.Series(True, index=df.index)

    out = df[filt].copy()

    # If specific classes were requested, use them (overrides include_reject)
    if allowed_classes is not None and "meta_class" in out.columns:
        out = out[out["meta_class"].isin(allowed_classes)].copy()
    else:
        # original behavior: exclude Reject unless explicitly included
        if not include_reject and "meta_class" in out.columns:
            out = out[out["meta_class"] != 0].copy()

    return out


def compute_trade_level_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Direction: +1 for buy, -1 for sell
    dir_map = {"buy": 1.0, "sell": -1.0}
    df["direction"] = df["trade_type"].str.lower().map(dir_map).fillna(0.0)

    # Ensure numeric
    for col in ["entry_price", "exit_price", "sl", "profit", "equity_base"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Price move in direction of trade
    df["price_move"] = (df["exit_price"] - df["entry_price"]) * df["direction"]

    # Risk per trade from SL distance (absolute). Use np.abs because np.where returns ndarray.
    # For buy: risk = entry - sl; for sell: risk = sl - entry
    risk_raw = np.where(
        df["direction"] >= 0,
        df["entry_price"] - df["sl"],
        df["sl"] - df["entry_price"],
    )
    df["risk_distance"] = np.abs(risk_raw)

    # Realized R multiple (signed). Guard div-by-zero / NaN.
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.divide(df["price_move"], df["risk_distance"])
    R = pd.to_numeric(R, errors="coerce")
    R.replace([np.inf, -np.inf], np.nan, inplace=True)
    df["R_multiple"] = R

    # Per-trade return relative to equity base
    df["return_pct"] = np.where(
        (df["equity_base"].notna()) & (df["equity_base"] > 0),
        df["profit"] / df["equity_base"],
        np.nan,
    )

    # Duration (minutes) – recompute from timestamps if available
    if "entry_time" in df.columns and "exit_time" in df.columns:
        et = pd.to_datetime(df["entry_time"], errors="coerce", utc=True)
        xt = pd.to_datetime(df["exit_time"], errors="coerce", utc=True)
        df["trade_duration_min"] = (xt - et).dt.total_seconds() / 60.0

    # Win/Loss flags
    df["is_win"] = df["profit"] > 0
    df["is_loss"] = df["profit"] < 0

    return df



def equity_curve(df: pd.DataFrame) -> pd.DataFrame:
    # Sort by exit time (fallback: entry_time, then index)
    sort_cols = [col for col in ["exit_time", "entry_time"] if col in df.columns]
    if not sort_cols:
        sort_cols = None
    df = df.sort_values(sort_cols or df.index.tolist()).copy()

    # Base equity start
    if df["equity_base"].notna().any():
        start_equity = df["equity_base"].dropna().iloc[0]
    else:
        # Fallback: start at 100 if unknown
        start_equity = 100.0

    df["equity"] = start_equity + df["profit"].cumsum()
    return df


def drawdown_stats(equity_series: pd.Series) -> Dict[str, Any]:
    roll_max = equity_series.cummax()
    dd = (equity_series - roll_max)
    dd_pct = (equity_series / roll_max) - 1.0
    max_dd = dd.min()  # negative value
    max_dd_pct = dd_pct.min()  # negative %
    return {
        "max_drawdown_abs": float(max_dd),
        "max_drawdown_pct": float(max_dd_pct),
    }


def consecutive_losses(s: pd.Series) -> int:
    # Max streak of losses
    max_streak = 0
    cur = 0
    for v in s:
        if v < 0:
            cur += 1
            max_streak = max(max_streak, cur)
        else:
            cur = 0
    return int(max_streak)


def payoff_ratio(df: pd.DataFrame) -> float:
    avg_win = df.loc[df["is_win"], "profit"].mean()
    avg_loss = df.loc[df["is_loss"], "profit"].mean()
    if np.isnan(avg_win) or np.isnan(avg_loss) or avg_loss == 0:
        return np.nan
    return float(avg_win / abs(avg_loss))


def expectancy_dollars(df: pd.DataFrame) -> float:
    p_win = df["is_win"].mean()
    avg_win = df.loc[df["is_win"], "profit"].mean()
    avg_loss = df.loc[df["is_loss"], "profit"].mean()
    avg_win = 0.0 if np.isnan(avg_win) else avg_win
    avg_loss = 0.0 if np.isnan(avg_loss) else avg_loss
    return float(p_win * avg_win + (1 - p_win) * avg_loss)


def expectancy_R(df: pd.DataFrame) -> float:
    wins = df.loc[df["R_multiple"] > 0, "R_multiple"]
    losses = df.loc[df["R_multiple"] < 0, "R_multiple"]
    p_win = (df["R_multiple"] > 0).mean()
    avg_win_R = wins.mean() if not wins.empty else 0.0
    avg_loss_R = losses.mean() if not losses.empty else 0.0
    return float(p_win * avg_win_R + (1 - p_win) * avg_loss_R)


def profit_factor(df: pd.DataFrame) -> float:
    gross_profit = df.loc[df["profit"] > 0, "profit"].sum()
    gross_loss = df.loc[df["profit"] < 0, "profit"].abs().sum()
    if gross_loss == 0:
        return np.inf if gross_profit > 0 else np.nan
    return float(gross_profit / gross_loss)


def annualized_metrics_from_daily(equity_daily: pd.Series) -> Dict[str, Any]:
    daily_returns = equity_daily.pct_change().dropna()
    if daily_returns.empty:
        return {
            "sharpe_annual": np.nan,
            "sortino_annual": np.nan,
            "cagr": np.nan,
            "vol_annual": np.nan,
        }

    mean = daily_returns.mean()
    std = daily_returns.std(ddof=1)
    downside = daily_returns[daily_returns < 0].std(ddof=1)

    sharpe = np.nan if std == 0 else (mean / std) * np.sqrt(252)
    sortino = np.nan if downside == 0 or np.isnan(downside) else (mean / downside) * np.sqrt(252)
    vol_annual = std * np.sqrt(252)

    # CAGR from first to last equity with calendar days
    start_val, end_val = equity_daily.iloc[0], equity_daily.iloc[-1]
    days = (equity_daily.index[-1] - equity_daily.index[0]).days or 1
    cagr = (end_val / start_val) ** (365.0 / days) - 1.0 if start_val > 0 else np.nan

    return {
        "sharpe_annual": float(sharpe),
        "sortino_annual": float(sortino),
        "cagr": float(cagr),
        "vol_annual": float(vol_annual),
    }
def drawdown_from_series(equity: pd.Series) -> Dict[str, float]:
    """Generic DD on any equity series (no resample)."""
    roll_max = equity.cummax()
    dd_abs = (equity - roll_max)
    dd_pct = (equity / roll_max) - 1.0
    return {
        "max_dd_abs_series": float(dd_abs.min()),
        "max_dd_pct_series": float(dd_pct.min()),
    }

def trade_level_risk_metrics(trade_returns: pd.Series) -> Dict[str, float]:
    """
    Sharpe/Sortino from per-trade returns (profit / equity_base).
    This is NOT time-annualized; it’s per-trade. If you want an annualized
    version, multiply by sqrt(trades_per_year). For now we report per-trade.
    """
    r = trade_returns.dropna()
    if r.empty:
        return {"sharpe_per_trade": np.nan, "sortino_per_trade": np.nan, "vol_per_trade": np.nan}
    mean = r.mean()
    std = r.std(ddof=1)
    downside = r[r < 0].std(ddof=1)
    sharpe = np.nan if (std == 0 or np.isnan(std)) else mean / std
    sortino = np.nan if (downside == 0 or np.isnan(downside)) else mean / downside
    return {
        "sharpe_per_trade": float(sharpe),
        "sortino_per_trade": float(sortino),
        "vol_per_trade": float(std),
    }

def per_class_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Small table by meta_class with counts, win rate, PF.
    meta_class: 0=Reject, 1=1:1, 2=1:2, 3=1:3 (your convention).
    """
    rows = []
    for c in sorted(df["meta_class"].dropna().unique()):
        sub = df[df["meta_class"] == c]
        if sub.empty:
            continue
        wins = sub["profit"] > 0
        gross_profit = sub.loc[sub["profit"] > 0, "profit"].sum()
        gross_loss = sub.loc[sub["profit"] < 0, "profit"].abs().sum()
        pf = np.inf if gross_loss == 0 and gross_profit > 0 else (gross_profit / gross_loss if gross_loss > 0 else np.nan)
        rows.append({
            "meta_class": int(c),
            "trades": int(len(sub)),
            "win_rate": float(wins.mean()),
            "profit_factor": float(pf),
            "avg_R": float(sub["R_multiple"].mean()),
            "expectancy_$": float(sub["profit"].mean()),
        })
    return pd.DataFrame(rows)

def normalize_class_args(vals):
    """
    Accepts ['1', '2'] or ['1:1','1:2'] etc., returns a set of ints {1,2}.
    Allowed: 0=Reject, 1=1:1, 2=1:2, 3=1:3
    """
    if not vals:
        return None
    mapping = {"0":"0", "1":"1", "2":"2", "3":"3", "1:1":"1", "1:2":"2", "1:3":"3", "reject":"0"}
    out = set()
    for v in vals:
        key = str(v).strip().lower()
        if key in mapping:
            out.add(int(mapping[key]))
        else:
            # try raw int
            try:
                out.add(int(key))
            except:
                pass
    return out if out else None


def main():
    ap = argparse.ArgumentParser(description="Analyze trade JSON and export metrics.")
    ap.add_argument("--trades", type=Path, required=True, help="Path to trades.json")
    ap.add_argument("--threshold", type=float, default=0.85, help="Meta approval threshold")
    ap.add_argument("--include-reject", action="store_true", help="Include trades with meta_class == 0 (Reject)")
    ap.add_argument("--outdir", type=Path, default=Path("."), help="Output directory")
    ap.add_argument(
    "--only-classes",
    nargs="+",
    help="Filter by meta classes; pass numbers (e.g. 1 2) or labels (e.g. 1:1 1:2). 0=Reject, 1=1:1, 2=1:2, 3=1:3",
    )
    args = ap.parse_args()
    allowed_classes = normalize_class_args(args.only_classes)


    df_raw = load_trades(args.trades)
    df_threshold = filter_trades(df_raw, args.threshold, include_reject=True, allowed_classes=None)
    df_filt = filter_trades(df_raw, args.threshold, include_reject=args.include_reject, allowed_classes=allowed_classes)
    df = compute_trade_level_metrics(df_filt)

    if df.empty:
        print("No trades after filtering. Check threshold/include-reject options.")
        return

    # Build equity curve (trade-level)
    df = equity_curve(df)
        # --- Trade-level drawdown (works even for 1 day) ---
    dd_trade = drawdown_from_series(df["equity"])

    # --- Trade-level Sharpe/Sortino from per-trade returns ---
    trade_risk = trade_level_risk_metrics(df["return_pct"])

    # --- Per-class breakdown (optional but super useful) ---
    class_table = per_class_breakdown(df)
    class_table.to_csv(args.outdir / "per_class_metrics.csv", index=False)

    impact_table = class_impact_report(df)
    impact_table.to_csv(args.outdir / "class_impact_report.csv", index=False)

    print("\n=== CLASS IMPACT REPORT ===")
    print(impact_table.to_string(index=False))


    class_table_all = per_class_breakdown(compute_trade_level_metrics(df_threshold))
    class_table_all.to_csv(args.outdir / "per_class_metrics_all.csv", index=False)


    # Daily equity (using exit_time, forward-fill within days)
    if "exit_time" in df.columns and df["exit_time"].notna().any():
        eq = df.set_index(df["exit_time"]).sort_index()["equity"]
    elif "entry_time" in df.columns and df["entry_time"].notna().any():
        eq = df.set_index(df["entry_time"]).sort_index()["equity"]
    else:
        # synthetic index
        eq = df["equity"].copy()
        eq.index = pd.date_range("2000-01-01", periods=len(eq), freq="D")

    equity_daily = eq.resample("D").last().ffill()
    dd = drawdown_stats(equity_daily)

    # Core summary metrics
    summary = {
        "trades": int(len(df)),
        "win_rate": float(df["is_win"].mean()),
        "avg_trade_minutes": float(df["trade_duration_min"].mean()),
        "gross_profit": float(df.loc[df["profit"] > 0, "profit"].sum()),
        "gross_loss": float(df.loc[df["profit"] < 0, "profit"].sum()),
        "net_profit": float(df["profit"].sum()),
        "profit_factor": profit_factor(df),
        "expectancy_$": expectancy_dollars(df),
        "expectancy_R": expectancy_R(df),
        "avg_R": float(df["R_multiple"].mean()),
        "median_R": float(df["R_multiple"].median()),
        "payoff_ratio": payoff_ratio(df),
        "max_consecutive_losses": consecutive_losses(df["profit"]),
        "max_drawdown_abs": dd["max_drawdown_abs"],
        "max_drawdown_pct": dd["max_drawdown_pct"],
    }

    # Kelly (based on R multiples)
    wins_R = df.loc[df["R_multiple"] > 0, "R_multiple"]
    losses_R = df.loc[df["R_multiple"] < 0, "R_multiple"]
    p = (df["R_multiple"] > 0).mean()
    avg_win_R = wins_R.mean() if not wins_R.empty else np.nan
    avg_loss_R = abs(losses_R.mean()) if not losses_R.empty else np.nan
    b = (avg_win_R / avg_loss_R) if (avg_loss_R and not np.isnan(avg_loss_R)) else np.nan
    kelly = np.nan if (b is np.nan or np.isnan(p)) else max(0.0, p - (1 - p) / b) if (b and b != 0 and not np.isnan(b)) else np.nan
    summary["kelly_fraction_est"] = float(kelly) if not np.isnan(kelly) else np.nan

    # Annualized risk/return stats from daily equity
    annual = annualized_metrics_from_daily(equity_daily)
    summary.update(annual)

    # MAR ratio (CAGR / |MaxDD%|)
    mar = np.nan
    if not np.isnan(summary["cagr"]) and summary["max_drawdown_pct"] != 0:
        mar = summary["cagr"] / abs(summary["max_drawdown_pct"])
    summary["mar_ratio"] = float(mar) if not np.isnan(mar) else np.nan
    summary["max_drawdown_abs_trade_level"] = dd_trade["max_dd_abs_series"]
    summary["max_drawdown_pct_trade_level"] = dd_trade["max_dd_pct_series"]

    # Add trade-level Sharpe/Sortino
    summary.update(trade_risk)

    # Save per-trade metrics
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    trade_cols = [
        "id", "symbol", "trade_type", "entry_time", "exit_time",
        "entry_price", "sl", "exit_price", "profit",
        "equity_base", "equity", "trade_duration_min",
        "meta_class", "meta_max_prob",
        "R_multiple", "return_pct",
        "is_win"
    ]
    trade_cols = [c for c in trade_cols if c in df.columns]
    df[trade_cols].to_csv(outdir / "trade_metrics.csv", index=False)

    # Save summary (single row)
    pd.DataFrame([summary]).to_csv(outdir / "summary_metrics.csv", index=False)

    # Also print a neat summary to console
    print("\n=== SUMMARY METRICS ===")
    for k, v in summary.items():
        if isinstance(v, float):
            if "pct" in k or k in {"sharpe_annual", "sortino_annual", "mar_ratio", "profit_factor"}:
                print(f"{k:>24}: {v:.4f}")
            else:
                print(f"{k:>24}: {v:.4f}")
        else:
            print(f"{k:>24}: {v}")
    print(f"\nSaved per-trade -> {outdir / 'trade_metrics.csv'}")
    print(f"Saved summary   -> {outdir / 'summary_metrics.csv'}")


if __name__ == "__main__":
    main()

#     python str.py --trades trades.json --threshold 0.85 --outdir results
# How to use
# Only 1:2 trades:

# python str.py --trades trades.json --threshold 0.85 --only-classes 2 --outdir results
# # or
# python str.py --trades trades.json --threshold 0.85 --only-classes 1:2 --outdir results
# Only 1:1 trades:

# bash
# Copy
# python str.py --trades trades.json --threshold 0.85 --only-classes 1 --outdir results
# # or
# python str.py --trades trades.json --threshold 0.85 --only-classes 1:1 --outdir results
# 1:1 and 1:2 together (exclude Reject by default):

# bash
# Copy
# python str.py --trades trades.json --threshold 0.85 --only-classes 1 2 --outdir results
