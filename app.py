# app.py
import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
from io import StringIO
import requests
import csv
import warnings

warnings.filterwarnings("ignore")

# -------------------------------
# 技術分析模組
# -------------------------------
def check_technical_signal(df, mode="standard"):
    if len(df) < 20:
        return None
    
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
    if curr['Volume'] < 200000:
        return None
    
    ma20 = df['Close'].rolling(20).mean().iloc[-1]
    ma60 = df['Close'].rolling(60).mean().iloc[-1]
    vol_ma5 = df['Volume'].rolling(5).mean().iloc[-1]
    
    pct_change = (curr['Close'] - prev['Close']) / prev['Close'] * 100
    vol_ratio = curr['Volume'] / vol_ma5 if vol_ma5 > 0 else 0
    
    if mode == "strong":
        if pct_change >= 3.0 and curr['Close'] > ma20:
            return f"🔥 強勢(+{pct_change:.1f}%) 爆量{vol_ratio:.1f}倍"
    else:
        if curr['Close'] > ma20 and curr['Close'] > ma60:
            if abs(ma20-ma60)/curr['Close'] < 0.05 and pct_change > 0.5:
                return "🚀 多頭排列 蓄勢"
    return None

# -------------------------------
# Streamlit 介面
# -------------------------------
st.title("📡 台股強勢股快篩 (V82 Hybrid Radar)")

stock_input = st.text_input("輸入股票代號 (空白隔開)", "2330 2454 2603 2881")
mode = st.radio("選擇模式", ["standard", "strong"])

if st.button("開始掃描"):
    stock_list = stock_input.split()
    tickers_tw = [f"{c}.TW" for c in stock_list]
    tickers_two = [f"{c}.TWO" for c in stock_list]
    
    # 嘗試下載上市與上櫃
    try:
        data = yf.download(" ".join(tickers_tw + tickers_two), period="3mo", group_by='ticker', auto_adjust=True)
    except Exception as e:
        st.error("❌ 無法下載行情資料，請檢查網路。")
        st.stop()
    
    results = []
    for code in stock_list:
        ticker_tw = f"{code}.TW"
        ticker_two = f"{code}.TWO"
        
        df = pd.DataFrame()
        if ticker_tw in data.columns.levels[0]:
            df = data[ticker_tw]
        elif ticker_two in data.columns.levels[0]:
            df = data[ticker_two]
        
        if not df.empty:
            df = df.dropna()
            sig = check_technical_signal(df, mode)
            if sig:
                price = df['Close'].iloc[-1]
                pct = (price - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100
                results.append({
                    "代號": code,
                    "現價": f"{price:.1f}",
                    "漲跌": f"{pct:+.1f}%",
                    "技術訊號": sig
                })
    
    if results:
        st.success(f"🎉 共找到 {len(results)} 檔潛力股")
        st.dataframe(pd.DataFrame(results))
    else:
        st.info("💡 今日無符合條件的股票")

