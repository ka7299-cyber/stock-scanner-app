import streamlit as st
import pandas as pd
import yfinance as yf
import requests, csv, datetime
from io import StringIO
import numpy as np
from scipy.signal import argrelextrema
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# 內建產業板塊資料庫 (取自 V82)
# ==========================================
SECTOR_DB = {
    '半導體/IC設計': ['2330','2454','2303','3711','3034','3035','2379','3443','3661','6531'],
    'AI伺服器/電腦': ['2382','3231','2356','6669','2376','2377','2301','2353','2324','2357'],
    '航運/運輸': ['2603','2609','2615','2618','2610','2637','2606','2605','2634','2636'],
    '金融/金控': ['2881','2882','2891','2886','2884','2892','2885','2880','2890','2883'],
    '生技/醫療': ['1795','6472','6446','6547','6589','4147','4174','4128','4736','4105'],
    '台股權值百大': ['2330','2317','2454','2382','2881','2412','2882','2308','2891','3711','2002']
}

# ==========================================
# 特定股票指定均線對照表 (None 代表自動計算)
# ==========================================
CUSTOM_MA_DB = {
    '1210': {'short': 26, 'long': 48}, '1216': {'short': 26, 'long': None},
    '1477': {'short': 25, 'long': None}, '1514': {'short': None, 'long': 51},
    '2006': {'short': 21, 'long': 48}, '2301': {'short': 18, 'long': 53},
    '2303': {'short': 21, 'long': 48}, '2308': {'short': 27, 'long': 97},
    '2313': {'short': 20, 'long': 61}, '2317': {'short': 18, 'long': 57},
    '2324': {'short': 19, 'long': 57}, '2327': {'short': 20, 'long': None},
    '2330': {'short': 17, 'long': 57}, '2337': {'short': 28, 'long': None},
    '2344': {'short': 31, 'long': None}, '2345': {'short': 28, 'long': 60},
    '2352': {'short': 19, 'long': 59}, '2353': {'short': 17, 'long': 63},
    '2354': {'short': 34, 'long': None}, '2356': {'short': 23, 'long': 58},
    '2357': {'short': 21, 'long': None}, '2360': {'short': 21, 'long': None},
    '2362': {'short': 23, 'long': None}, '2368': {'short': 22, 'long': 60},
    '2376': {'short': None, 'long': 29}, '2377': {'short': 18, 'long': None},
    '2379': {'short': 26, 'long': None}, '2382': {'short': 23, 'long': 57},
    '2383': {'short': 18, 'long': 50}, '2385': {'short': 20, 'long': 55},
    '2395': {'short': 22, 'long': 49}, '2404': {'short': 29, 'long': None},
    '2408': {'short': 23, 'long': 43}, '2409': {'short': 18, 'long': 52},
    '2428': {'short': 24, 'long': 59}, '2439': {'short': 18, 'long': 74},
    '2454': {'short': 29, 'long': 60}, '2472': {'short': 24, 'long': 48},
    '2496': {'short': 29, 'long': None}, '2603': {'short': 35, 'long': None},
    '2727': {'short': 28, 'long': 46}, '2753': {'short': 29, 'long': 52},
    '2755': {'short': 22, 'long': 53}, '2891': {'short': 18, 'long': 47}, 
    '3005': {'short': 21, 'long': 62}, '3017': {'short': 21, 'long': 55}, 
    '3029': {'short': 20, 'long': 43}, '3036': {'short': 18, 'long': 53}, 
    '3130': {'short': 22, 'long': 36}, '3231': {'short': 26, 'long': 76}, 
    '3443': {'short': 18, 'long': 67}, '3583': {'short': 21, 'long': 50}, 
    '3706': {'short': 23, 'long': None}, '4987': {'short': 25, 'long': None}, 
    '5904': {'short': 21, 'long': 57}, '6138': {'short': 21, 'long': None}, 
    '6146': {'short': 25, 'long': 67}, '6176': {'short': 22, 'long': None}, 
    '6191': {'short': 25, 'long': None}, '6192': {'short': 29, 'long': None}, 
    '6197': {'short': 23, 'long': 48}, '6201': {'short': 34, 'long': None}, 
    '6239': {'short': 23, 'long': 48}, '6279': {'short': 19, 'long': 53}, 
    '6284': {'short': 24, 'long': 56}, '6285': {'short': 21, 'long': 59}, 
    '6409': {'short': 23, 'long': 50}, '6667': {'short': 26, 'long': 44}, 
    '6669': {'short': 28, 'long': 58}, '6721': {'short': 17, 'long': 41}, 
    '6728': {'short': 30, 'long': 48}, '6805': {'short': 18, 'long': None}, 
    '8210': {'short': 25, 'long': None}, '8367': {'short': 22, 'long': 55}, 
    '9939': {'short': 17, 'long': 57}, '1519': {'short': 25, 'long': 45},
}

# ==========================================
# 均線最佳化演算法
# ==========================================
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
# ==========================================
# 籌碼模組 (簡化版 V160)
# ==========================================
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

def analyze_chip_status(m, i, s, trend, vol_ratio=1.0):
    tags = "🟢 籌碼中性"
    if i and s:
        f_buy = i[0]
        t_buy = i[1]
        sbl_chg = s[1]
        m_chg = m[1] if m else 0

        if f_buy > 0 and vol_ratio >= 2.0:
            return "🚀 外資帶量突破", "🔥 強勢多頭 (抱緊)"
        if t_buy > 0 and vol_ratio >= 2.0:
            return "🔥 投信帶量鎖股", "🔥 強勢多頭 (抱緊)"
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
    
# ==========================================
# 參考 Sniper-X 樣式的詳細圖表顯示函式
# ==========================================
def show_single_stock_detail(stock_id):
    st.subheader(f"📊 股票代號：{stock_id} 詳細技術與籌碼分析")
    
    # 抓取 K 線行情
    t_symbol = f"{stock_id}.TW"
    df = yf.Ticker(t_symbol).history(period="1y")
    if df.empty:
        df = yf.Ticker(f"{stock_id}.TWO").history(period="1y")
        
    if df.empty:
        st.error(f"❌ 找不到股票代號 {stock_id} 的數據")
        return

    # 計算均線
    custom_cfg = CUSTOM_MA_DB.get(stock_id, {})
    short_ma = custom_cfg.get('short') or find_best_ma_v2(df, 16, 25)
    long_ma = custom_cfg.get('long') or find_best_ma_v2(df, 45, 70)
    
    df['MS'] = df['Close'].rolling(window=short_ma).mean()
    df['ML'] = df['Close'].rolling(window=long_ma).mean()
    
    # 抓取籌碼
    crawler = ChipCrawlerV160(stock_id)
    target_date = df.index[-1].to_pydatetime().date()
    m, i, s = crawler.get_latest_chip_summary(target_date)
    
    last = df.iloc[-1]
    price = last['Close']
    ms_v = last['MS']
    ml_v = last['ML']
    
    # 1. 頂部 KPI 卡片
    c1, c2, c3 = st.columns(3)
    c1.metric("現價", f"{price:.2f}")
    c2.metric(f"短均線 ({short_ma}日)", f"{ms_v:.2f}")
    c3.metric(f"長均線 ({long_ma}日)", f"{ml_v:.2f}")
    
    # 2. 當日籌碼數據卡片
    st.markdown("### 🔍 當日籌碼數據詳情")
    f_buy = i[0] if i else 0
    t_buy = i[1] if i else 0
    m_bal, m_chg = (m[0], m[1]) if m else (0, 0)
    sbl_bal, sbl_chg = (s[0], s[1]) if s else (0, 0)
    
    cols = st.columns(4)
    with cols[0]:
        st.write(f"**外資買賣超**：{f_buy:+d} 張")
    with cols[1]:
        st.write(f"**投信買賣超**：{t_buy:+d} 張")
    with cols[2]:
        st.write(f"**融資變化**：{m_chg:+d} 張 (餘額: {m_bal})")
    with cols[3]:
        st.write(f"**借券變化**：{sbl_chg:+d} 張 (餘額: {sbl_bal:,})")
        
# 3. K 線與量能圖表 (參考 Sniper-X-V150 專業設定)
    # 取最近 120 日資料 (可在 30~240 日間自由調整，例如改 60, 120, 240)
    p_df = df.tail(120).copy()
    
    # 強制將索引轉為標準日期格式，解決 2006/2009 年分顯示異常
    p_df['Date_Str'] = pd.to_datetime(p_df.index).strftime('%Y-%m-%d')
    
    # 建立雙層圖表 (上: K線與均線, 下: 成交量)
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03, 
        row_width=[0.3, 0.7]
    )
    
    # K 棒繪製
    fig.add_trace(go.Candlestick(
        x=p_df['Date_Str'], 
        open=p_df['Open'], 
        high=p_df['High'], 
        low=p_df['Low'], 
        close=p_df['Close'],
        name='K棒', 
        increasing_line_color='#ef5350', 
        decreasing_line_color='#26a69a'
    ), row=1, col=1)
    
    # 短/長均線繪製
    fig.add_trace(go.Scatter(
        x=p_df['Date_Str'], y=p_df['MS'], mode='lines', 
        name=f'短均({short_ma}日)', line=dict(color='orange', width=1.5)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=p_df['Date_Str'], y=p_df['ML'], mode='lines', 
        name=f'長均({long_ma}日)', line=dict(color='blue', width=1.5)
    ), row=1, col=1)
    
    # 成交量柱狀圖
    colors = ['#ef5350' if c >= o else '#26a69a' for c, o in zip(p_df['Close'], p_df['Open'])]
    fig.add_trace(go.Bar(
        x=p_df['Date_Str'], y=p_df['Volume']/ 1000, 
        name='成交量(張)', marker_color=colors
    ), row=2, col=1)
    
    # 核心設定 1：強制類別軸 (剔除非交易日/假日中斷) + 貫穿式十字游標
    fig.update_xaxes(
        type='category',             # 強制連續，無交易日不留空白
        spikecolor="gray",          # 游標輔助線顏色
        spikethickness=1,           # 游標粗細
        spikemode="across",         # 游標線貫穿上下雙圖表
        spikesnap="cursor", 
        showspikes=True             # 開啟貫穿對齊線
    )
    
# 核心設定 2：隱藏 RangeSlider + 將 Hover 數據統一顯示在頂部固定抬頭區
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=550, 
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",       # 將開高低收與成交量數據統一顯示在頂部橫條卡片中
        hoverlabel=dict(
            bgcolor="rgba(255, 255, 255, 0.95)", # 白色背景，避免遮擋
            font_size=12
        ),
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
# ==========================================
# Streamlit 主介面
# ==========================================
# 判斷是否網址帶有 stock 參數 (在新分頁開啟時觸發)
show_stock_id = st.query_params.get("stock")
if show_stock_id:
    show_single_stock_detail(show_stock_id)
    st.stop() # 停止執行下方快篩介面
st.title("📡 台股強勢股快篩 (V160 均線+籌碼)")

option = st.radio("選擇模式", ["自選股票", "全市場強勢股"])

if option == "自選股票":
    stock_input = st.text_input("輸入股票代號 (空白隔開)", "2330 2454 2603")
    stock_list = stock_input.split()
else:
    # 全市場強勢股：撈取所有代號
    all_codes = []
    for sector in SECTOR_DB.values():
        all_codes += sector
    stock_list = list(set(all_codes))

# 初始化 results 變數，確保未按按鈕前不會跳出 NameError
results = []

if st.button("開始掃描"):
    tickers = [f"{c}.TW" for c in stock_list]

    try:
        data = yf.download(" ".join(tickers), period="6mo", group_by='ticker', auto_adjust=True)
    except Exception as e:
        st.error("❌ 無法下載行情資料，請檢查網路。")
        st.stop()

    for code in stock_list:
        ticker = f"{code}.TW"
        if ticker not in data.columns.levels[0]:
            continue
        df = data[ticker].dropna()
        if len(df) < 70: 
            continue

        # 均線計算 (優先讀取指定參數，若為 None 則自動計算)
        custom_cfg = CUSTOM_MA_DB.get(code, {})
        short_ma = custom_cfg.get('short') or find_best_ma_v2(df, 16, 25)
        long_ma = custom_cfg.get('long') or find_best_ma_v2(df, 45, 70)
        df['MS'] = df['Close'].rolling(window=short_ma).mean()
        df['ML'] = df['Close'].rolling(window=long_ma).mean()

        last = df.iloc[-1]
        price = last['Close']
        ms_v = last['MS']
        ml_v = last['ML']
        pct = (price - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100

       # ========== 新增：成交量計算與比較 ==========
        volume_today = df['Volume'].iloc[-1]                      # 當日成交量
        avg_volume_5d = df['Volume'].iloc[-6:-1].mean()          # 前 5 日平均成交量 (不含當日)
        
        # 計算爆量比例 (當日量 / 5日均量)
        vol_ratio = (volume_today / avg_volume_5d) if avg_volume_5d > 0 else 0
        
        # 當日量與 5 日均量比較標籤
        if vol_ratio >= 2.0:
            vol_status = f"🔥 爆量 ({vol_ratio:.1f}倍)"
        elif vol_ratio >= 1.2:
            vol_status = f"📈 增量 ({vol_ratio:.1f}倍)"
        elif vol_ratio <= 0.7:
            vol_status = f"📉 縮量 ({vol_ratio:.1f}倍)"
        else:
            vol_status = f"➡️ 平量 ({vol_ratio:.1f}倍)"
        # ============================================

        # 趨勢判斷
        if price > ms_v and ms_v > ml_v:
            # ★ 全市場強勢股條件：漲幅 >3%
            if option == "全市場強勢股" and pct < 3:
                continue
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

        crawler = ChipCrawlerV160(code)
        # 轉為 Python 標準 date 物件，避開時區與時間字串格式干擾
        target_date = df.index[-1].to_pydatetime().date()
        m, i, s = crawler.get_latest_chip_summary(target_date)
        # 帶入 vol_ratio 進行分析
        chip_msg, trend = analyze_chip_status(m, i, s, trend, vol_ratio=vol_ratio)
        
        results.append({
            "代號": f'<a href="/?stock={code}" target="_blank">{code}</a>',
            "現價": f"{price:.1f}",
            "漲跌": f"{pct:+.1f}%",
            "當日成交量(張)": f"{int(volume_today / 1000):,}",      # 轉為千張/手
            "5日均量(張)": f"{int(avg_volume_5d / 1000):,}",
            "成交量變化": vol_status,                             # 觀察比較結果
            "短均線": f"{ms_v:.1f}",
            "長均線": f"{ml_v:.1f}",
            "趨勢判斷": trend,
            "籌碼分析": chip_msg
        })

# 結果輸出
if results:
    st.success(f"🎉 共找到 {len(results)} 檔標的")
    df_out = pd.DataFrame(results)
    st.write(df_out.to_html(escape=False, index=False), unsafe_allow_html=True)
elif st.session_state.get("scanned", False):  # 避免剛進網頁就顯示警告
    st.warning("⚠️ 查無符合條件的股票")
