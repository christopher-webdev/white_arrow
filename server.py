# server.py
import os, time, json, csv
from collections import defaultdict, deque
from datetime import datetime
from typing import Any, Dict, List

from fastapi import FastAPI, Depends, HTTPException, Request, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
from datetime import timezone
import math
from fastapi import Query

import numpy as np
import pandas as pd


load_dotenv()

API_PREFIX = os.getenv("API_PREFIX", "/api")
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "change-me")
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")]
TRADES_FILE = os.getenv("TRADES_FILE", "trades_gbp.json")
PRED_DIR = os.getenv("PRED_DIR", "latest_predictions")
SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", "GBPUSD+,USDCAD+").split(",")]

app = FastAPI(title="White-Arrow API (file-mode)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if ALLOWED_ORIGINS == ["*"] else ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

security = HTTPBearer()


def _sanitize_json(obj):
    """Recursively replace NaN/Inf with None so JSON serialization never fails."""
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_json(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    return obj

def require_auth(cred: HTTPAuthorizationCredentials = Depends(security)):
    if cred.credentials != AUTH_TOKEN:
        raise HTTPException(401, "Invalid token")
    return True

# ---- Tiny in-memory rate limiter (per IP, per path) ----
WINDOW_SECONDS = 10
MAX_CALLS = 30
_calls = defaultdict(deque)  # type: ignore[var-annotated]

@app.middleware("http")
async def rate_limit(request: Request, call_next):
    ip = request.client.host if request.client else "unknown"
    key = f"{ip}:{request.url.path}"
    now = time.time()
    dq = _calls[key]
    dq.append(now)
    while dq and now - dq[0] > WINDOW_SECONDS:
        dq.popleft()
    if len(dq) > MAX_CALLS:
        raise HTTPException(429, "Too many requests")
    return await call_next(request)

# ----------------- Helpers -----------------

def _to_iso(ts: str | None) -> str | None:
    if not ts:
        return None
    try:
        # normalize: if no offset, treat as UTC
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()
    except Exception:
        return None

def _safe_float(x) -> float:
    try:
        return float(x)
    except:
        return 0.0

def _load_trades_from_json(path: str) -> List[Dict[str, Any]]:
    """Load bot trades.json (id->trade) and normalize to list."""
    if not os.path.exists(path):
        return []
    data = json.load(open(path, "r", encoding="utf-8"))
    rows: List[Dict[str, Any]] = []
    for _, t in data.items():
        r = dict(t)
        # flatten classifier_conf
        cc = t.get("classifier_conf") or {}
        for k, v in cc.items():
            r[k] = v
        # meta max prob (meta_classifier: [reject, 1:1, 1:2, 1:3])
        mc = t.get("meta_classifier") or []
        if isinstance(mc, list) and mc:
            r["meta_max_prob"] = max(mc)
        # normalize times
        r["entry_time"] = _to_iso(r.get("entry_time"))
        r["exit_time"] = _to_iso(r.get("exit_time"))
        # normalize numerics
        r["profit"] = _safe_float(r.get("profit"))
        r["equity_at_entry"] = _safe_float(r.get("equity_at_entry"))
        r["balance_at_entry"] = _safe_float(r.get("balance_at_entry"))
        r["is_win"] = r["profit"] > 0
        rows.append(r)
    return rows

def _compute_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {...}

    def sort_key(r):
        t = r.get("exit_time") or r.get("entry_time")
        if not t:
            return float("-inf")
        try:
            dt = datetime.fromisoformat(t)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()  # numeric, avoids aware/naive compare
        except Exception:
            return float("-inf")

    rows_sorted = sorted(rows, key=sort_key)

    gross_p = sum(r["profit"] for r in rows_sorted if r["profit"] > 0)
    gross_l = sum(-r["profit"] for r in rows_sorted if r["profit"] < 0)
    net = gross_p - gross_l
    wins = sum(1 for r in rows_sorted if r["profit"] > 0)
    losses = sum(1 for r in rows_sorted if r["profit"] < 0)
    wr = wins / (wins + losses) if (wins + losses) else 0.0
    pf = (gross_p / gross_l) if gross_l > 0 else (None if gross_p == 0 else float("inf"))

    # base equity
    base = None
    for r in rows_sorted:
        eb = r.get("equity_at_entry") or r.get("balance_at_entry")
        if eb and eb > 0:
            base = eb
            break
    if base is None:
        base = 100.0

    equity_curve = []
    running = base
    for r in rows_sorted:
        running += r["profit"]
        t = r.get("exit_time") or r.get("entry_time")
        equity_curve.append({"t": t, "equity": running})

    per_class: Dict[int, Dict[str, Any]] = {}
    for r in rows_sorted:
        c = r.get("meta_class")
        if c is None:
            continue
        c = int(c)
        g = per_class.setdefault(c, {"trades": 0, "wins": 0, "gp": 0.0, "gl": 0.0})
        g["trades"] += 1
        if r["profit"] > 0:
            g["wins"] += 1
            g["gp"] += r["profit"]
        elif r["profit"] < 0:
            g["gl"] += -r["profit"]

    per_class_rows = []
    for c, v in sorted(per_class.items()):
        pc_pf = (v["gp"] / v["gl"]) if v["gl"] > 0 else (None if v["gp"] == 0 else float("inf"))
        per_class_rows.append({
            "meta_class": c,
            "trades": v["trades"],
            "win_rate": v["wins"] / v["trades"] if v["trades"] else 0.0,
            "profit_factor": pc_pf
        })

    return {
        "trades": len(rows_sorted),
        "win_rate": wr,
        "profit_factor": pf,
        "gross_profit": gross_p,
        "gross_loss": gross_l,
        "net_profit": net,
        "equity_curve": equity_curve,
        "per_class": per_class_rows
    }

def _pred_csv_path(symbol: str) -> str:
    return os.path.join(PRED_DIR, f"{symbol}.csv")

def _read_predictions(symbol: str, limit: int = 200) -> List[Dict[str, Any]]:
    """Read latest_predictions/<symbol>.csv without pandas."""
    path = _pred_csv_path(symbol)
    if not os.path.exists(path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["timestamp"] = _to_iso(row.get("timestamp"))
            for k in ("clf_1_1_prob", "clf_1_2_prob", "reg_pred", "meta_class", "accepted"):
                if k in row and row[k] not in (None, ""):
                    try:
                        row[k] = float(row[k]) if k != "meta_class" else int(float(row[k]))
                    except:
                        pass
            rows.append(row)
    rows = rows[-limit:]
    rows.sort(key=lambda r: r.get("timestamp") or "", reverse=True)
    return rows
# ---- Full metrics using pandas (mirrors analyze_trades.py) ----
def _rows_to_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    # timestamps
    for col in ("entry_time","exit_time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    # equity base
    if "equity_at_entry" not in df: df["equity_at_entry"] = np.nan
    if "balance_at_entry" not in df: df["balance_at_entry"] = np.nan
    df["equity_base"] = df["equity_at_entry"].fillna(df["balance_at_entry"])
    # normalize
    if "trade_type" in df:
        df["trade_type"] = df["trade_type"].str.lower()
    for col in ("entry_price","exit_price","sl","profit"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # direction
    dir_map = {"buy":1.0, "sell":-1.0}
    df["direction"] = df.get("trade_type","").map(dir_map).fillna(0.0)
    # price move and risk distance
    df["price_move"] = (df["exit_price"] - df["entry_price"]) * df["direction"]
    risk_raw = np.where(df["direction"] >= 0, df["entry_price"]-df["sl"], df["sl"]-df["entry_price"])
    df["risk_distance"] = np.abs(risk_raw)
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.divide(df["price_move"], df["risk_distance"])
    R = pd.to_numeric(R, errors="coerce").replace([np.inf,-np.inf], np.nan)
    df["R_multiple"] = R
    # returns
    df["return_pct"] = np.where((df["equity_base"].notna()) & (df["equity_base"]>0), df["profit"]/df["equity_base"], np.nan)
    # duration
    if "entry_time" in df and "exit_time" in df:
        dt = (df["exit_time"]-df["entry_time"]).dt.total_seconds()/60.0
        df["trade_duration_min"] = dt
    df["is_win"]  = df["profit"] > 0
    df["is_loss"] = df["profit"] < 0
    # meta max prob if present
    if "meta_max_prob" not in df and "meta_classifier" in df:
        df["meta_max_prob"] = df["meta_classifier"].apply(lambda x: max(x) if isinstance(x, list) and x else np.nan)
    return df

def _equity_curve(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [c for c in ["exit_time","entry_time"] if c in df.columns]
    df = df.sort_values(sort_cols or df.index.tolist()).copy()
    start = df["equity_base"].dropna().iloc[0] if df["equity_base"].notna().any() else 100.0
    df["equity"] = start + df["profit"].cumsum()
    return df

def _profit_factor(df: pd.DataFrame) -> float:
    gp = df.loc[df["profit"]>0,"profit"].sum()
    gl = df.loc[df["profit"]<0,"profit"].abs().sum()
    if gl == 0: return float("inf") if gp>0 else np.nan
    return float(gp/gl)

def _drawdown_stats(equity: pd.Series) -> dict:
    roll_max = equity.cummax()
    dd_abs = equity - roll_max
    dd_pct = (equity/roll_max) - 1.0
    return {"max_drawdown_abs": float(dd_abs.min()), "max_drawdown_pct": float(dd_pct.min())}

def _consec_losses(pnl: pd.Series) -> int:
    max_streak = cur = 0
    for v in pnl:
        if v < 0: cur += 1; max_streak = max(max_streak, cur)
        else: cur = 0
    return int(max_streak)

def _per_class(df: pd.DataFrame) -> list[dict]:
    out = []
    if "meta_class" not in df: return out
    for c, sub in df.groupby("meta_class"):
        gp = sub.loc[sub["profit"]>0,"profit"].sum()
        gl = sub.loc[sub["profit"]<0,"profit"].abs().sum()
        pf = float("inf") if gl==0 and gp>0 else (float(gp/gl) if gl>0 else np.nan)
        out.append({
            "meta_class": int(c) if pd.notna(c) else None,
            "trades": int(len(sub)),
            "win_rate": float((sub["profit"]>0).mean()),
            "profit_factor": pf,
            "avg_R": float(sub["R_multiple"].mean()),
            "expectancy_$": float(sub["profit"].mean()),
        })
    return out

def _full_metrics(rows: list[dict], threshold: float=0.0, include_reject: bool=False, only_classes: list[int]|None=None) -> dict:
    df = _rows_to_df(rows)
    # filter by meta prob threshold
    if "meta_max_prob" in df.columns and threshold>0:
        df = df[df["meta_max_prob"] >= threshold]
    # class filter
    if only_classes is not None and "meta_class" in df.columns:
        df = df[df["meta_class"].isin(only_classes)]
    elif not include_reject and "meta_class" in df.columns:
        df = df[df["meta_class"] != 0]
    if df.empty:
        return {"summary": {"trades": 0}, "equity_curve": [], "per_class": [], "trades_table": []}

    df = _equity_curve(df)

    # daily equity for CAGR/Sharpe/Sortino
    if "exit_time" in df and df["exit_time"].notna().any():
        eq = df.set_index(df["exit_time"]).sort_index()["equity"]
    elif "entry_time" in df and df["entry_time"].notna().any():
        eq = df.set_index(df["entry_time"]).sort_index()["equity"]
    else:
        eq = df["equity"].copy()
        eq.index = pd.date_range("2000-01-01", periods=len(eq), freq="D")
    equity_daily = eq.resample("D").last().ffill()

    # summary metrics (match your script)
    gp = float(df.loc[df["profit"]>0,"profit"].sum())
    gl = float(df.loc[df["profit"]<0,"profit"].sum())
    net = float(df["profit"].sum())
    ev = float((df["is_win"].mean() * df.loc[df["is_win"],"profit"].mean()) + ((1-df["is_win"].mean()) * df.loc[df["is_loss"],"profit"].mean()))
    wins_R = df.loc[df["R_multiple"]>0,"R_multiple"]
    losses_R = df.loc[df["R_multiple"]<0,"R_multiple"]
    p = float((df["R_multiple"]>0).mean())
    avg_win_R = float(wins_R.mean()) if not wins_R.empty else 0.0
    avg_loss_R = float(losses_R.mean()) if not losses_R.empty else 0.0
    expectancy_R = float(p*avg_win_R + (1-p)*avg_loss_R)
    payoff = float(df.loc[df["is_win"],"profit"].mean() / abs(df.loc[df["is_loss"],"profit"].mean())) if df["is_loss"].any() else np.nan
    dd_trade = _drawdown_stats(df["equity"])
    max_consec = _consec_losses(df["profit"])

    # annualized metrics
    daily_ret = equity_daily.pct_change().dropna()
    mean = daily_ret.mean(); std = daily_ret.std(ddof=1); downside = daily_ret[daily_ret<0].std(ddof=1)
    sharpe = float((mean/std)*np.sqrt(252)) if std and not np.isnan(std) and std!=0 else np.nan
    sortino = float((mean/downside)*np.sqrt(252)) if downside and not np.isnan(downside) and downside!=0 else np.nan
    start_val, end_val = equity_daily.iloc[0], equity_daily.iloc[-1]
    days = max(1, (equity_daily.index[-1]-equity_daily.index[0]).days)
    cagr = float((end_val/start_val)**(365.0/days)-1.0) if start_val>0 else np.nan
    mar = float(cagr/abs(dd_trade["max_drawdown_pct"])) if cagr==cagr and dd_trade["max_drawdown_pct"]!=0 else np.nan

    # Kelly (R-based)
    b = (avg_win_R/abs(avg_loss_R)) if avg_loss_R!=0 else np.nan
    kelly = float(max(0.0, p - (1-p)/b)) if b==b and b not in (0.0, np.nan) else np.nan

    # per-class table
    per_cls = _per_class(df)

    # trade table (trim)
    keep = ["id","symbol","trade_type","entry_time","exit_time","entry_price","sl","exit_price","profit","equity_base","equity","trade_duration_min","meta_class","meta_max_prob","R_multiple","return_pct","is_win"]
    keep = [c for c in keep if c in df.columns]
    trades_table = df[keep].sort_values("exit_time").tail(500).fillna("").to_dict(orient="records")

    summary = {
        "trades": int(len(df)),
        "win_rate": float(df["is_win"].mean()),
        "avg_trade_minutes": float(df["trade_duration_min"].mean()) if "trade_duration_min" in df else np.nan,
        "gross_profit": gp,
        "gross_loss": gl,
        "net_profit": net,
        "profit_factor": _profit_factor(df),
        "expectancy_$": ev,
        "expectancy_R": expectancy_R,
        "avg_R": float(df["R_multiple"].mean()),
        "median_R": float(df["R_multiple"].median()),
        "payoff_ratio": payoff,
        "max_consecutive_losses": max_consec,
        "max_drawdown_abs": dd_trade["max_drawdown_abs"],
        "max_drawdown_pct": dd_trade["max_drawdown_pct"],
        "sharpe_annual": sharpe,
        "sortino_annual": sortino,
        "cagr": cagr,
        "mar_ratio": mar,
        "kelly_fraction_est": kelly,
    }

    equity_curve = [{"t": (t.tz_convert("UTC") if hasattr(t,"tz_convert") else t).isoformat() if pd.notna(t) else None, "equity": float(v)} for t,v in eq.items()]
    return {"summary": summary, "equity_curve": equity_curve, "per_class": per_cls, "trades_table": trades_table}

# ----------------- Routes -----------------
api = APIRouter(prefix=API_PREFIX, tags=["api"])

@api.get("/health")
def health():
    return {"status": "ok"}

@api.get("/symbols", dependencies=[Depends(require_auth)])
def symbols():
    found = []
    if os.path.isdir(PRED_DIR):
        for name in os.listdir(PRED_DIR):
            if name.lower().endswith(".csv"):
                found.append(name[:-4])
    uniq = sorted(set(found or SYMBOLS))
    return {"symbols": uniq}

@api.get("/trades", dependencies=[Depends(require_auth)])
def get_trades():
    rows = _load_trades_from_json(TRADES_FILE)
    rows.sort(key=lambda r: r.get("exit_time") or r.get("entry_time") or "", reverse=True)
    return rows

@api.get("/metrics", dependencies=[Depends(require_auth)])
def get_metrics():
    rows = _load_trades_from_json(TRADES_FILE)
    return _compute_metrics(rows)

@api.get("/predictions/{symbol}", dependencies=[Depends(require_auth)])
def get_predictions(symbol: str, limit: int = 200):
    if not symbol:
        raise HTTPException(400, "symbol required")
    return _read_predictions(symbol, limit=limit)

@app.post(f"{API_PREFIX}/auth/register")
def register_disabled():
    raise HTTPException(403, "Registration disabled")


@api.get("/metrics", dependencies=[Depends(require_auth)])
def get_metrics():
    rows = _load_trades_from_json(TRADES_FILE)
    data = _compute_metrics(rows)
    return _sanitize_json(data)

@api.get("/metrics/full", dependencies=[Depends(require_auth)])
def metrics_full(
    threshold: float = Query(0.0, ge=0.0, le=1.0),
    include_reject: bool = Query(False),
    only_classes: str | None = Query(None)
):
    rows = _load_trades_from_json(TRADES_FILE)
    only = [int(x) for x in only_classes.split(",")] if only_classes else None
    data = _full_metrics(rows, threshold=threshold, include_reject=include_reject, only_classes=only)
    return _sanitize_json(data)
app.include_router(api)
