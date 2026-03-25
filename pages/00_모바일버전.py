import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ── 페이지 설정 ─────────────────────────────────────
st.set_page_config(
    page_title="주식 분석 대시보드",
    page_icon="📈",
    layout="wide", # 모바일에서도 화면을 넓게 쓰도록 필수 적용
)

st.title("📈 국내 · 글로벌 주식 분석 대시보드")
st.caption("yfinance 기반 · 실시간 데이터 | 📱 모바일 화면 최적화 적용")

# [모바일 유저를 위한 안내 문구 추가]
st.info("💡 스마트폰 화면에서는 화면 왼쪽 상단의 **'>' 버튼**을 눌러 메뉴를 열고 종목을 선택하세요!")

# ── 종목 정의 ────────────────────────────────────────
KOREAN_STOCKS = {
    "삼성전자":    "005930.KS",
    "SK하이닉스":  "000660.KS",
    "LG에너지솔루션": "373220.KS",
    "현대차":      "005380.KS",
    "POSCO홀딩스": "005490.KS",
    "카카오":      "035720.KS",
    "NAVER":       "035420.KS",
    "셀트리온":    "068270.KS",
    "KB금융":      "105560.KS",
    "기아":        "000270.KS",
}

GLOBAL_STOCKS = {
    "Apple":      "AAPL",
    "Microsoft":  "MSFT",
    "NVIDIA":     "NVDA",
    "Tesla":      "TSLA",
    "Amazon":     "AMZN",
    "Alphabet":   "GOOGL",
    "Meta":       "META",
    "TSMC":       "TSM",
    "ASML":       "ASML",
    "Berkshire":  "BRK-B",
}

ALL_STOCKS = {**KOREAN_STOCKS, **GLOBAL_STOCKS}

# ── 사이드바 설정 ────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 설정")

    market = st.radio("시장 선택", ["국내", "글로벌", "전체 비교"])

    period_map = {
        "1개월": "1mo", "3개월": "3mo",
        "6개월": "6mo", "1년": "1y", "2년": "2y",
    }
    period_label = st.select_slider("기간", list(period_map.keys()), value="1년")
    period = period_map[period_label]

    if market == "국내":
        stock_pool = KOREAN_STOCKS
    elif market == "글로벌":
        stock_pool = GLOBAL_STOCKS
    else:
        stock_pool = ALL_STOCKS

    selected_names = st.multiselect(
        "종목 선택",
        list(stock_pool.keys()),
        default=list(stock_pool.keys())[:5],
    )

    analysis = st.selectbox(
        "분석 유형",
        [
            "📊 수익률 비교 (정규화)",
            "🕯️ 캔들차트 + 이동평균",
            "🌡️ 상관관계 히트맵",
            "⚡ 변동성 분석",
            "📉 RSI 지표",
            "🗺️ 버블차트 (시총·PER·배당)",
            "📦 수익률 분포 (박스플롯)",
        ],
    )

    st.divider()
    st.caption("데이터: Yahoo Finance")

if not selected_names:
    st.warning("왼쪽 메뉴에서 관심 있는 종목을 하나 이상 선택해 주세요.")
    st.stop()

selected_tickers = {n: stock_pool[n] for n in selected_names}

# ── 데이터 로드 ──────────────────────────────────────
@st.cache_data(ttl=600)
def load_prices(tickers: dict, period: str) -> pd.DataFrame:
    symbols = list(tickers.values())
    raw = yf.download(symbols, period=period, auto_adjust=True, progress=False)
    close = raw["Close"] if len(symbols) > 1 else raw["Close"].to_frame(symbols[0])
    close.columns = [k for k, v in tickers.items() if v in close.columns]
    return close.dropna(how="all")


@st.cache_data(ttl=600)
def load_info(tickers: dict) -> dict:
    info = {}
    for name, sym in tickers.items():
        try:
            t = yf.Ticker(sym)
            info[name] = t.info
        except Exception:
            info[name] = {}
    return info


with st.spinner("📡 데이터를 불러오는 중입니다... 잠시만 기다려주세요!"):
    prices = load_prices(selected_tickers, period)
    available = [n for n in selected_names if n in prices.columns]
    prices = prices[available]

if prices.empty:
    st.error("데이터를 가져올 수 없습니다. 종목 또는 기간을 확인해 주세요.")
    st.stop()

# ── 공통 유틸 ────────────────────────────────────────
COLORS = px.colors.qualitative.Bold

def returns(df: pd.DataFrame) -> pd.DataFrame:
    return df.pct_change().dropna()

# ════════════════════════════════════════════════════
# 분석 1 · 수익률 비교 (정규화)
# ════════════════════════════════════════════════════
if analysis == "📊 수익률 비교 (정규화)":
    st.subheader("📊 누적 수익률 비교 (시작일 = 100)")

    norm = prices / prices.iloc[0] * 100
    fig = px.line(norm, labels={"value": "지수", "variable": "종목"},
                  color_discrete_sequence=COLORS)
    # [모바일 팁] margin 값으로 상하좌우 여백을 줄여 좁은 화면 활용도 극대화
    fig.update_layout(hovermode="x unified", height=480,
                      legend=dict(orientation="h", y=-0.15),
                      margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_
