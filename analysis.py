"""
投資成績分析システムの中核ロジックを実装したPythonモジュール。

主な責務:
- 約定履歴（fills）を仕様書に従って1トレード=1行へ正規化
- pnl_pctとRを算出し、基本指標や条件別期待値を集計
- J-Quantsの日足・指数データを取り込み、特徴量を付与

pandas DataFrameを入力・出力として扱い、Notebookやバッチ処理どちらでも利用できるように設計。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Literal, Optional, Sequence

import numpy as np
import pandas as pd

Side = Literal["Buy", "Sell"]


class TradeNormalizationError(ValueError):
    """約定データが仕様違反の場合に送出される例外。"""


@dataclass
class TradeResult:
    trade_id: int
    code: str
    entry_date: datetime
    exit_date: datetime
    avg_entry_price: float
    avg_exit_price: float
    buy_amount: float
    sell_amount: float
    fee_amount: float
    pnl_yen: float
    pnl_pct: float
    r_multiple: float
    holding_days: int

    def as_dict(self) -> dict:
        return {
            "trade_id": self.trade_id,
            "code": self.code,
            "entry_date": self.entry_date.date(),
            "exit_date": self.exit_date.date(),
            "avg_entry_price": self.avg_entry_price,
            "avg_exit_price": self.avg_exit_price,
            "buy_amount": self.buy_amount,
            "sell_amount": self.sell_amount,
            "fee_amount": self.fee_amount,
            "pnl_yen": self.pnl_yen,
            "pnl_pct": self.pnl_pct,
            "R": self.r_multiple,
            "holding_days": self.holding_days,
        }


def _validate_fills(df: pd.DataFrame) -> pd.DataFrame:
    """仕様の必須列とサイドを検証し、ソート用インデックスを付与。"""

    required = {
        "trade_fill_id",
        "code",
        "side",
        "exec_datetime",
        "price",
        "quantity",
        "fee",
    }
    missing = required - set(df.columns)
    if missing:
        raise TradeNormalizationError(f"欠損列があります: {sorted(missing)}")

    side_ok = {"Buy", "Sell"}
    invalid_side = set(df["side"]) - side_ok
    if invalid_side:
        raise TradeNormalizationError(f"不正なsideがあります: {sorted(invalid_side)}")

    if (df["quantity"] <= 0).any():
        raise TradeNormalizationError("quantityは正の数である必要があります")

    if not np.issubdtype(df["exec_datetime"].dtype, np.datetime64):
        df = df.copy()
        df["exec_datetime"] = pd.to_datetime(df["exec_datetime"])

    df = df.copy()
    df["_input_order"] = np.arange(len(df))
    return df


def normalize_trades(fills: pd.DataFrame) -> pd.DataFrame:
    """
    約定履歴を1トレード=1行へ正規化してトレードDataFrameを返す。

    ルール:
    - exec_datetime昇順で処理し、同一日時は入力順を優先。
    - Buyで残高を積み上げ、Sellで残高0になった時点で1トレード成立。
    - pnl_pct, Rは仕様の固定損切り8%に対して算出。

    Parameters
    ----------
    fills : pd.DataFrame
        trade_fill_id, code, side, exec_datetime, price, quantity, fee を含む約定データ。

    Returns
    -------
    pd.DataFrame
        trade_id, code, entry_date, exit_date, avg_entry_price, avg_exit_price,
        buy_amount, sell_amount, fee_amount, pnl_yen, pnl_pct, R, holding_days を列にもつ。
    """

    df = _validate_fills(fills)
    df = df.sort_values(["exec_datetime", "_input_order"]).reset_index(drop=True)

    trades: List[TradeResult] = []
    active: dict[str, dict] = {}
    next_trade_id = 1

    for _, row in df.iterrows():
        code = row["code"]
        side: Side = row["side"]
        qty = float(row["quantity"])
        price = float(row["price"])
        fee = float(row.get("fee", 0.0))
        exec_dt: datetime = row["exec_datetime"]

        state = active.get(code)

        if side == "Sell" and state is None:
            raise TradeNormalizationError(f"Sell先行を検知しました: code={code}, exec_datetime={exec_dt}")

        if side == "Buy":
            if state is None:
                state = {
                    "trade_id": next_trade_id,
                    "code": code,
                    "buy_qty": 0.0,
                    "sell_qty": 0.0,
                    "buy_amount": 0.0,
                    "sell_amount": 0.0,
                    "fee_amount": 0.0,
                    "entry_date": exec_dt,
                    "last_exit": exec_dt,
                }
                next_trade_id += 1
                active[code] = state

            state["buy_qty"] += qty
            state["buy_amount"] += price * qty
            state["fee_amount"] += fee
            state["entry_date"] = min(state["entry_date"], exec_dt)
        else:  # Sell
            if state is None:
                raise TradeNormalizationError(f"Sell処理時にポジションがありません: code={code}")

            new_qty = state["buy_qty"] - state["sell_qty"] - qty
            if new_qty < -1e-9:
                raise TradeNormalizationError(
                    f"残高がマイナスになります: code={code}, exec_datetime={exec_dt}, quantity={qty}"
                )

            state["sell_qty"] += qty
            state["sell_amount"] += price * qty
            state["fee_amount"] += fee
            state["last_exit"] = exec_dt

            if abs(new_qty) < 1e-9:  # ポジション完結
                avg_entry_price = state["buy_amount"] / state["buy_qty"]
                avg_exit_price = state["sell_amount"] / state["sell_qty"] if state["sell_qty"] else 0.0
                pnl_yen = state["sell_amount"] - state["buy_amount"] - state["fee_amount"]
                pnl_pct = pnl_yen / state["buy_amount"] if state["buy_amount"] else 0.0
                r_multiple = pnl_pct / 0.08  # 損切り-8%を基準にしたR
                holding_days = (state["last_exit"].date() - state["entry_date"].date()).days

                trades.append(
                    TradeResult(
                        trade_id=state["trade_id"],
                        code=code,
                        entry_date=state["entry_date"],
                        exit_date=state["last_exit"],
                        avg_entry_price=avg_entry_price,
                        avg_exit_price=avg_exit_price,
                        buy_amount=state["buy_amount"],
                        sell_amount=state["sell_amount"],
                        fee_amount=state["fee_amount"],
                        pnl_yen=pnl_yen,
                        pnl_pct=pnl_pct,
                        r_multiple=r_multiple,
                        holding_days=holding_days,
                    )
                )
                active.pop(code, None)

    if active:
        open_codes = ", ".join(sorted(active.keys()))
        raise TradeNormalizationError(f"クローズしていないトレードがあります: {open_codes}")

    trade_df = pd.DataFrame([t.as_dict() for t in trades])
    if not trade_df.empty:
        trade_df = trade_df.sort_values(["entry_date", "trade_id"]).reset_index(drop=True)
    return trade_df


def compute_basic_metrics(trades: pd.DataFrame) -> pd.DataFrame:
    """トレード全体の件数・勝率・平均リターン・平均Rなど基本指標を返す。"""
    if trades.empty:
        return pd.DataFrame([
            {
                "trades": 0,
                "win_rate": np.nan,
                "avg_return_pct": np.nan,
                "avg_R": np.nan,
                "max_losing_streak": np.nan,
            }
        ])

    wins = (trades["pnl_pct"] > 0).sum()
    losing_streak = _compute_max_losing_streak(trades["pnl_pct"].to_list())

    return pd.DataFrame(
        [
            {
                "trades": len(trades),
                "win_rate": wins / len(trades),
                "avg_return_pct": trades["pnl_pct"].mean(),
                "avg_R": trades["R"].mean(),
                "max_losing_streak": losing_streak,
            }
        ]
    )


def _compute_max_losing_streak(pnl_list: Sequence[float]) -> int:
    max_streak = 0
    current = 0
    for pnl in pnl_list:
        if pnl <= 0:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak


def evaluate_by_conditions(trades: pd.DataFrame, condition_columns: Iterable[str]) -> pd.DataFrame:
    """
    条件（特徴量）別に平均Rなどを集計するユーティリティ。

    Parameters
    ----------
    trades : pd.DataFrame
        normalize_tradesで生成したDataFrameに特徴量列を追加したもの。
    condition_columns : Iterable[str]
        グルーピング対象の列名。複数指定するとクロス集計になる。
    """
    if trades.empty:
        return pd.DataFrame()

    grouped = trades.groupby(list(condition_columns))
    summary = grouped.agg(
        trades=("trade_id", "count"),
        avg_R=("R", "mean"),
        win_rate=("pnl_pct", lambda s: (s > 0).mean()),
        avg_return_pct=("pnl_pct", "mean"),
    )
    return summary.reset_index().sort_values("avg_R", ascending=False)


def attach_market_features(
    trades: pd.DataFrame,
    market: pd.DataFrame,
    index_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    エントリー日に紐づく特徴量を付与する。

    - 52週高値更新フラグ: closeが過去252営業日のmaxを更新
    - 出来高急増率: 当日出来高 ÷ 過去20日平均
    - ギャップ率: 当日始値/前日終値 - 1
    - 地合いフラグ: index_dfのcloseが25MAより上ならTrue
    """
    if trades.empty:
        return trades

    mkt = market.copy()
    mkt["date"] = pd.to_datetime(mkt["date"])
    mkt = mkt.sort_values(["code", "date"]).reset_index(drop=True)

    mkt["prev_close"] = mkt.groupby("code")["close"].shift(1)
    mkt["gap_rate"] = (mkt["open"] / mkt["prev_close"]) - 1
    mkt["vol_ma20"] = mkt.groupby("code")["volume"].transform(lambda s: s.rolling(20, min_periods=1).mean())
    mkt["volume_surge"] = mkt["volume"] / mkt["vol_ma20"]
    mkt["high_52w"] = mkt.groupby("code")["close"].transform(lambda s: s.rolling(252, min_periods=1).max())
    mkt["is_52w_high"] = mkt["close"] >= mkt["high_52w"]

    features = mkt[["code", "date", "gap_rate", "volume_surge", "is_52w_high"]]

    if index_df is not None and not index_df.empty:
        idx = index_df.copy()
        idx["date"] = pd.to_datetime(idx["date"])
        idx = idx.sort_values("date")
        if "ma_25" not in idx:
            idx["ma_25"] = idx["close"].rolling(25, min_periods=1).mean()
        idx["trend_flag"] = idx["close"] >= idx["ma_25"]
        features = features.merge(idx[["date", "trend_flag"]], on="date", how="left")

    trades = trades.copy()
    trades["entry_date"] = pd.to_datetime(trades["entry_date"])
    enriched = trades.merge(
        features,
        left_on=["code", "entry_date"],
        right_on=["code", "date"],
        how="left",
    ).drop(columns=["date"], errors="ignore")

    return enriched


def demo_pipeline(fills: pd.DataFrame, market: pd.DataFrame, index_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    仕様書の処理フローを1本の関数で実行するサンプル。

    1. 約定データをトレード単位に正規化
    2. 市場データとマージして特徴量付与
    3. 基本指標を別途算出可能
    """
    trades = normalize_trades(fills)
    trades_with_features = attach_market_features(trades, market, index_df=index_df)
    return trades_with_features


if __name__ == "__main__":
    # 簡易デモ用のダミーデータセット
    fills_demo = pd.DataFrame(
        [
            {
                "trade_fill_id": 1,
                "code": "7203",
                "side": "Buy",
                "exec_datetime": "2024-01-05 09:00",
                "price": 1000,
                "quantity": 100,
                "fee": 50,
            },
            {
                "trade_fill_id": 2,
                "code": "7203",
                "side": "Sell",
                "exec_datetime": "2024-01-10 10:00",
                "price": 1080,
                "quantity": 100,
                "fee": 50,
            },
        ]
    )

    market_demo = pd.DataFrame(
        [
            {"date": "2024-01-04", "code": "7203", "open": 990, "high": 1010, "low": 980, "close": 995, "volume": 120000},
            {"date": "2024-01-05", "code": "7203", "open": 995, "high": 1015, "low": 990, "close": 1005, "volume": 150000},
            {"date": "2024-01-10", "code": "7203", "open": 1060, "high": 1090, "low": 1050, "close": 1085, "volume": 180000},
        ]
    )

    index_demo = pd.DataFrame(
        [
            {"date": "2024-01-04", "index": "TOPIX", "close": 2400},
            {"date": "2024-01-05", "index": "TOPIX", "close": 2410},
            {"date": "2024-01-10", "index": "TOPIX", "close": 2450},
        ]
    )

    trades = demo_pipeline(fills_demo, market_demo, index_df=index_demo)
    print("トレード一覧:\n", trades)
    print("基本指標:\n", compute_basic_metrics(trades))
