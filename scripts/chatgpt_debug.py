#!/usr/bin/env python3
"""Simple debug script using ChatGPT to read news and simulate trades."""
import json
import os
import time
from datetime import datetime
from typing import List

import requests
from colorama import Fore, Style, init

try:
    import openai
except Exception:  # pragma: no cover - optional dependency
    openai = None  # type: ignore

CONFIG_PATH = "config_examples/config_chatgpt.example.json"
WHALE_PATH = "freqtrade/templates/whale_moves_sample.json"


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fetch_news(sources: List[str], max_sources: int = 5) -> List[str]:
    """Fetch news snippets from provided RSS sources."""
    articles: List[str] = []
    for src in sources[:max_sources]:
        try:
            response = requests.get(src, timeout=5)
            if response.status_code == 200:
                articles.append(response.text[:1000])
        except Exception:
            continue
    return articles


def load_whale_score(path: str) -> float:
    if not os.path.isfile(path):
        return 0.0
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not data:
        return 0.0
    return float(data[-1].get("whale_score", 0.0))


def analyze_news(articles: List[str], model: str, api_key: str) -> float:
    if openai is None:
        return 0.0
    openai.api_key = api_key
    prompt = (
        "Summarize the following news and provide a sentiment score between -1 and 1."
    )
    content = "\n".join(articles)
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": f"{prompt}\n{content}"}],
            timeout=10,
        )
        text = response["choices"][0]["message"]["content"].strip().split()[0]
        return float(text)
    except Exception as exc:  # pragma: no cover - network
        print(Fore.RED + f"OpenAI error: {exc}" + Style.RESET_ALL)
        return 0.0


def mock_trade(symbol: str, score: float) -> None:
    if score > 0.5:
        print(Fore.GREEN + f"BUY {symbol}" + Style.RESET_ALL)
    elif score < -0.5:
        print(Fore.RED + f"SELL {symbol}" + Style.RESET_ALL)
    else:
        print(Fore.YELLOW + f"HOLD {symbol}" + Style.RESET_ALL)


def main() -> None:
    init(autoreset=True)
    config = load_config(CONFIG_PATH)
    api_key = config["openai_api_key"]
    model = config.get("openai_model", "gpt-4")
    sources = config.get("news_sources", [])

    while True:
        news = fetch_news(sources)
        sentiment = analyze_news(news, model, api_key)
        whale_score = load_whale_score(WHALE_PATH)
        score = sentiment + whale_score
        print(f"{datetime.utcnow().isoformat()} sentiment: {score:.2f}")
        mock_trade("BTC/USDT", score)
        time.sleep(60)


if __name__ == "__main__":
    main()
