import streamlit as st
import pandas as pd
import yfinance as yf
import requests, csv, datetime
from io import StringIO
import numpy as np
from scipy.signal import argrelextrema

# -------------------------------
# 均線最佳化演算法
# -------------------------------
def find_best_ma_v2(df, start_day, end_day):
    closes = df['Close'].values
    lows = df['Low'].values
    best_ma = start_day
    best_score = -np.inf
    for ma_len in range(start_day, end_day + 1):
        ma = df['Close'].rolling(window=ma_len).mean().values
        valid = slice(ma_len, len(df))
        min_idxs = argrelextrema(lows[valid], np.less, order=3)[0]
        if len(min_idxs) == 0: continue
        err = (np.abs(lows[valid][min_idxs] - ma[valid][min_idxs]) / ma[valid][min_idxs]).mean()
        score = 100 - (err * 3000) + (ma_len - start_day) * 0.8
        if score > best_score:
            best_score = score
            best_ma = ma_len
    return best_ma

# -------------------------------
# 籌碼模組 (簡化版)
# -------------------------------
class ChipCrawlerV160:
    def __init__(self, stock_id, is_otc=False):
        self.stock_id = str(stock_id).strip()
        self.is_otc = is_otc
        self.headers = {'User-Agent': 'Mozilla/5.0'}

    def get_latest_chip_summary(self, target_date):
        m = self._get_margin(target_date)
        i = self._get_inst(target_date)
        s = self._get_sbl(target_date)
        return m, i, s

    def _get_margin(self, d):
        url = f"https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN?date={d.strftime('%Y%m%d')}&selectType=ALL&response=csv"
        try:
            r = requests.get(url, headers=self.headers, timeout=3)
            c = r.content.decode('big5', errors='ignore')
            reader = csv.reader(StringIO(c))
            for row in reader:
                if len(row)>13 and self.stock_id in row[0]:
                    def cl(v): return int(v.replace(',', '').strip())
                    return cl(row[6]), (cl(row[6])-cl(row[5])), cl(row[12]), (cl(row[12])-cl(row[11]))
        except: pass
        return None

    def _get_inst(self, d):
        url = f"https://www.twse.com.tw/rwd/zh/fund/T86?date={d.strftime('%Y%m%d')}&selectType=ALL&response=json"
        try:
            data = requests.get(url, headers=self.headers, timeout=3).json()
            if data['stat']=='OK':
                for rec in data['data']:
                    if rec[0] == self.stock_id:
                        def cl(v): return int(v.replace(',', '').strip())//1000
                        return cl(rec[4]), cl(rec[10]), cl(rec[11])
        except: pass
        return None

    def _get_sbl(self, d):
        url = f"https://www.twse.com.tw/rwd/zh/marginTrading/TWT93U?date={d.strftime('%Y%m%d')}&response=csv"
        try:
            c = requests.get(url, headers=self.headers, timeout=3).content.decode('big5', errors='ignore')
            reader = csv.reader(StringIO(c))
            for row in reader:
                if len(row)>12 and self.stock_id in row[0]:
                    def cl(v): return int(v.replace(',', '').strip())//1000
                    return cl(row[12]), (cl(row[12])-cl(row[8]))
        except: pass
        return None

def analyze_chip_status(m, i, s, trend):
    tags = "🟢 籌碼中性"
    if i and s:
        f_buy = i[0]
        t_buy = i[1]
        sbl_chg = s[1]
        m_chg = m[1] if m else 0

        if f_buy > 0 and sbl_chg < 0:
            tags = "🚀 外資真買"
        elif f_buy > 0 and sbl_chg > 200:
            tags = "⚠️ 外資假買"
        elif t_buy >= 3:
            tags = f"🔥 投信鎖股"
        elif f_buy >= 3:
            tags = f"💰 外資波盤"
        elif m_chg >= 3 and ("空頭" in trend or "轉弱" in trend):
            tags = "💀 散戶接刀"
        elif t_buy > 400:
            tags = "🚀 投信大買"
        elif f_buy > 800:
            tags = "💰 外資敲進"
        elif m_chg > 400 and f_buy < -400:
            tags = "📉 主力出貨"

        if ("多頭" in trend or "買點" in trend) and (t_buy > 0 or f_buy > 0) and "假買" not in tags:
            trend = "🏆 雙刀流：多頭確認"
        elif ("空頭" in trend or "轉弱" in trend) and (m_chg > 0 or sbl_chg > 0):
            trend = "💀 雙刀流：空頭警報"

    return tags, trend

# -------------------------------
# Streamlit 主介面
# -------------------------------
st.title("📡 台股強勢股快篩 (V160 均線+籌碼)")

stock_input = st.text_input("輸入股票代號 (空白隔開)", "2330 2454 2603")

if st.button("開始掃描"):
    stock_list = stock_input.split()
    tickers = [f"{c}.TW" for c in stock_list]

    try:
        data = yf.download(" ".join(tickers), period="6mo", group_by='ticker', auto_adjust=True)
    except Exception as e:
        st.error("❌ 無法下載行情資料，請檢查網路。")
        st.stop()

    results = []
    for code in stock_list:
        ticker = f"{code}.TW"
        if ticker not in data.columns.levels[0]:
            continue
        df = data[ticker].dropna()
        if len(df) < 70: continue

        # 均線計算
        short_ma = find_best_ma_v2(df, 16, 25)
        long_ma = find_best_ma_v2(df, 45, 70)
        df['MS'] = df['Close'].rolling(window=short_ma).mean()
        df['ML'] = df['Close'].rolling(window=long_ma).mean()

        last = df.iloc[-1]
        price = last['Close']
        ms_v = last['MS']
        ml_v = last['ML']

        # 趨勢判斷
        if price > ms_v and ms_v > ml_v:
            trend = "🔥 強勢多頭 (抱緊)"
        elif ms_v >= price >= ml_v:
            trend = "⚠️ 多頭回檔 (買點)"
        elif ms_v >= ml_v >= price:
            trend = "⚡ 跌破防線 (轉弱)"
        elif price > ms_v and ms_v <= ml_v:
            trend = "🛡️ 底部反彈 (搶短)"
        elif ml_v >= ms_v >= price:
            trend = "❄️ 絕對空頭 (觀望)"
        else:
            trend = "🧩 均線糾結 (震盪)"

        # 籌碼分析
        crawler = ChipCrawlerV160(code)
        m, i, s = crawler.get_latest_chip_summary(df.index[-1])
        chip_msg, trend = analyze_chip_status(m, i, s, trend)

        results.append({
            "代號": code,
            "現價": f"{price:.1f}",
            "短均線": f"{ms_v:.1f}",
            "長均線": f"{ml_v:.1f}",
            "趨勢判斷": trend,
            "籌碼分析": chip_msg
        })

    if results:
        st.success(f"🎉 共找到 {len(results)} 檔潛力股")
        st.dataframe(pd.DataFrame(results))
    else:
        st.info("💡 今日無符合條件的股票")
