# pragma pylint: disable=missing-docstring, invalid-name
from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import pandas as pd
from pandas import DataFrame

from freqtrade.strategy import IStrategy
from freqtrade.strategy.parameters import DecimalParameter

try:
    import openai
except Exception:  # pragma: no cover - optional dependency
    openai = None  # type: ignore


class ChatGPTSentimentStrategy(IStrategy):
    """Example strategy using ChatGPT sentiment as indicator."""

    INTERFACE_VERSION = 3

    # Strategy configuration
    timeframe = "1h"
    can_short: bool = True

    minimal_roi = {"0": 0.0}
    stoploss = -0.2

    # Hyperoptable thresholds
    long_threshold = DecimalParameter(0.0, 1.0, default=0.7, space="buy")
    short_threshold = DecimalParameter(-1.0, 0.0, default=-0.7, space="sell")

    startup_candle_count: int = 0

    def __init__(self) -> None:
        super().__init__()
        self.sentiment_df: Optional[pd.DataFrame] = None
        self.whale_df: Optional[pd.DataFrame] = None
        self.current_sentiment: Optional[float] = None
        self.last_sentiment_fetch: Optional[datetime] = None
        self.sentiment_path = os.path.join(
            "user_data",
            "data",
            "chatgpt_sentiment.json",
        )
        self.whale_path = os.path.join(
            "user_data",
            "data",
            "whale_moves.json",
        )
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4")

    def bot_start(self, **kwargs: Any) -> None:
        """Load sentiment data once on startup for backtesting/hyperopt."""
        runmode = self.config.get("runmode")
        if runmode and runmode.value in ("backtest", "hyperopt"):
            if os.path.isfile(self.sentiment_path):
                self.sentiment_df = pd.read_json(self.sentiment_path)
                if "date" in self.sentiment_df.columns:
                    self.sentiment_df["date"] = pd.to_datetime(self.sentiment_df["date"])
            if os.path.isfile(self.whale_path):
                self.whale_df = pd.read_json(self.whale_path)
                if "date" in self.whale_df.columns:
                    self.whale_df["date"] = pd.to_datetime(self.whale_df["date"])

    def _fetch_live_sentiment(self) -> Optional[float]:
        """Fetch latest sentiment from OpenAI API."""
        if openai is None:
            return None

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None

        openai.api_key = api_key
        prompt = "Provide current market sentiment as a number between -1 and 1"
        try:
            response = openai.ChatCompletion.create(
                model=self.openai_model,
                messages=[{"role": "user", "content": prompt}],
                timeout=10,
            )
            text = response["choices"][0]["message"]["content"].strip()
            return float(text)  # Expect the model to return a numeric sentiment
        except Exception:
            return None

    def bot_loop_start(self, current_time: datetime, **kwargs: Any) -> None:
        """Fetch new sentiment at most once per hour in live modes."""
        runmode = self.config.get("runmode")
        if runmode and runmode.value in ("live", "dry_run"):
            if (
                self.last_sentiment_fetch is None
                or current_time - self.last_sentiment_fetch > timedelta(hours=1)
            ):
                value = self._fetch_live_sentiment()
                if value is not None:
                    self.current_sentiment = value
                    self.last_sentiment_fetch = current_time

    def populate_indicators(self, dataframe: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        runmode = self.config.get("runmode")
        if runmode and runmode.value in ("backtest", "hyperopt"):
            if self.sentiment_df is not None:
                dataframe = pd.merge(
                    dataframe,
                    self.sentiment_df[["date", "sentiment"]],
                    on="date",
                    how="left",
                )
                dataframe["sentiment"].fillna(method="ffill", inplace=True)
            if self.whale_df is not None:
                dataframe = pd.merge(
                    dataframe,
                    self.whale_df[["date", "whale_score"]],
                    on="date",
                    how="left",
                )
                dataframe["whale_score"].fillna(0, inplace=True)
        else:
            dataframe["sentiment"] = self.current_sentiment
            dataframe["whale_score"] = 0.0

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0

        score = dataframe["sentiment"] + dataframe["whale_score"]

        dataframe.loc[
            score > self.long_threshold.value,
            "enter_long",
        ] = 1

        dataframe.loc[
            score < self.short_threshold.value,
            "enter_short",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0

        score = dataframe["sentiment"] + dataframe["whale_score"]

        dataframe.loc[
            score < 0,
            "exit_long",
        ] = 1

        dataframe.loc[
            score > 0,
            "exit_short",
        ] = 1

        return dataframe
