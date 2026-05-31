import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
import requests, csv
from io import StringIO

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
# 籌碼模組
# -------------------------------
class ChipCrawlerV82:
    def __init__(self, stock_id, is_otc=False):
        self.stock_id = str(stock_id).strip()
        self.is_otc = is_otc
        self.headers = {'User-Agent': 'Mozilla/5.0'}

    def get_latest_chip_summary(self, target_date):
        m = self._get_margin_retry(target_date)
        i = self._get_inst_retry(target_date)
        s = self._get_sbl_retry(target_date)
        return m, i, s

    def _get_margin_retry(self, date_obj):
        curr = date_obj
        for _ in range(3):
            res = self._get_twse_margin_csv(curr) if not self.is_otc else None
            if res: return res
            curr -= datetime.timedelta(days=1)
        return None

    def _get_inst_retry(self, date_obj):
        curr = date_obj
        for _ in range(3):
            res = self._get_twse_inst_json(curr) if not self.is_otc else None
            if res: return res
            curr -= datetime.timedelta(days=1)
        return None

    def _get_sbl_retry(self, date_obj):
        curr = date_obj
        for _ in range(3):
            res = self._get_twse_sbl_csv(curr) if not self.is_otc else None
            if res: return res
            curr -= datetime.timedelta(days=1)
        return None

    def _get_twse_margin_csv(self, d):
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

    def _get_twse_inst_json(self, d):
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

    def _get_twse_sbl_csv(self, d):
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

def analyze_chip_status(m, i, s):
    tags = []
    if i and i[1] > 50: tags.append("🔥投信買")
    if i and s:
        f_buy = i[0]
        sbl_chg = s[1]
        if f_buy > 100 and sbl_chg < 0: tags.append("🚀外資真買")
        elif f_buy > 100 and sbl_chg > 100: tags.append("⚠️外資假買")
    if m and m[1] > 200: tags.append("💀資增")
    if m and m[1] < -100: tags.append("✅資減")
    if m and m[3] > 100 and m[1] < 0: tags.append("📈軋空")
    return " ".join(tags) if tags else "籌碼平淡"

# -------------------------------
# Streamlit 主介面
# -------------------------------
st.title("📡 台股強勢股快篩 (技術 + 籌碼)")

stock_input = st.text_input("輸入股票代號 (空白隔開)", "2330 2454 2603")
mode = st.radio("選擇模式", ["standard", "strong"])

if st.button("開始掃描"):
    stock_list = stock_input.split()
    tickers = [f"{c}.TW" for c in stock_list]
    
    try:
        data = yf.download(" ".join(tickers), period="3mo", group_by='ticker', auto_adjust=True)
    except Exception as e:
        st.error("❌ 無法下載行情資料，請檢查網路。")
        st.stop()
    
    results = []
    for code in stock_list:
        ticker = f"{code}.TW"
        if ticker not in data.columns.levels[0]:
            continue
        df = data[ticker].dropna()
        sig = check_technical_signal(df, mode)
        if sig:
            crawler = ChipCrawlerV82(code)
            m, i, s = crawler.get_latest_chip_summary(df.index[-1])
            chip_msg = analyze_chip_status(m, i, s)

            price = df['Close'].iloc[-1]
            pct = (price - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100

            results.append({
                "代號": code,
                "現價": f"{price:.1f}",
                "漲跌": f"{pct:+.1f}%",
                "技術訊號": sig,
                "籌碼分析": chip_msg
            })

    if results:
        st.success(f"🎉 共找到 {len(results)} 檔潛力股")
        st.dataframe(pd.DataFrame(results))
    else:
        st.info("💡 今日無符合條件的股票")
