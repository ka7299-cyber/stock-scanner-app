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

CN_NAME_MAP = {
    "2330": "台積電", "2317": "鴻海", "2454": "聯發科", "2382": "廣達", "3231": "緯創",
    "2603": "長榮", "2609": "陽明", "2615": "萬海", "1519": "華城", "1504": "東元",
    "2356": "英業達", "2376": "技嘉", "3037": "欣興", "2303": "聯電", "3711": "日月光投控",
    "6669": "緯穎", "3661": "世芯-KY", "3443": "創意", "8046": "南電", "3035": "智原",
    "2881": "富邦金", "2882": "國泰金", "2891": "中信金", "2886": "兆豐金", "2884": "玉山金",
    "1795": "美時", "6472": "保瑞", "6446": "藥華藥", "2002": "中鋼", "2412": "中華電",
    "2344": "華邦電", "2408": "南亞科", "2337": "旺宏", "2308": "台達電", "2324": "仁寶",
    "2357": "華碩", "2301": "光寶科", "2377": "微星", "2353": "宏碁", "3008": "大立光",
    "2327": "國巨", "2492": "華新科", "3034": "聯詠", "2379": "瑞昱", "6531": "愛普*",
    "3529": "力旺", "6415": "矽力*-KY", "3017": "奇鋐", "3324": "雙鴻", "6230": "超眾",
    "6274": "台燿", "2383": "台光電", "6213": "聯茂", "3189": "景碩", "2368": "金像電",
    "2360": "致茂", "6409": "旭隼", "5269": "祥碩", "6643": "M31", "3680": "家登",
    "3131": "弘塑", "3583": "辛耘", "6187": "萬潤", "2409": "友達", "3481": "群創",
    "6116": "彩晶", "2340": "台亞", "2374": "佳能", "2354": "鴻準", "2352": "佳世達",
    "2312": "金寶", "2323": "中洋光", "3702": "大聯大", "3036": "文曄", "2347": "聯強",
    "2345": "智邦", "5388": "中磊", "6285": "啟碁", "2314": "揚智", "6202": "盛群"
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

def find_best_ma_golden_bluff_v2(df, start_day, end_day):
    closes = df['Close'].values; lows = df['Low'].values; highs = df['High'].values
    n = len(df)
    best_ma = start_day; best_score = -np.inf
    fib_nums = {21, 34, 55, 89}
    for ma_len in range(start_day, end_day + 1):
        ma_series = df['Close'].rolling(window=ma_len).mean()
        ma_values = ma_series.values
        if n < ma_len + 10: continue
        valid_idx = slice(ma_len, n)
        l_slice = lows[valid_idx]; h_slice = highs[valid_idx]; ma_slice = ma_values[valid_idx]
        min_idxs = argrelextrema(l_slice, np.less, order=3)[0]
        max_idxs = argrelextrema(h_slice, np.greater, order=3)[0]
        total_error = 0; point_count = 0
        if len(min_idxs) > 0:
            total_error += (np.abs(l_slice[min_idxs] - ma_slice[min_idxs]) / ma_slice[min_idxs]).sum()
            point_count += len(min_idxs)
        if len(max_idxs) > 0:
            total_error += (np.abs(h_slice[max_idxs] - ma_slice[max_idxs]) / ma_slice[max_idxs]).sum()
            point_count += len(max_idxs)
        avg_error = (total_error / point_count) if point_count > 0 else 0.05
        score = 100 - (avg_error * 3000) + (ma_len - start_day) * 0.8
        cross_mask = (closes[valid_idx] > ma_slice) ^ (np.roll(closes[valid_idx], 1) > np.roll(ma_slice, 1))
        if np.sum(cross_mask[1:]) / (len(ma_slice) / 20.0) > 3.0: score -= 100
        if ma_len in fib_nums: score += 10
        if score > best_score:
            best_score = score; best_ma = ma_len
    return best_ma

# ==========================================
# 籌碼模組 (支援近 5 日累積)
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

    def get_multi_day_summary(self, dates, lookback_days=5):
        f_sum, t_sum, m_sum = 0, 0, 0
        valid_days = 0
        for d in reversed(dates):
            if valid_days >= lookback_days:
                break
            i = self._get_inst(d)
            m = self._get_margin(d)
            if i or m:
                valid_days += 1
                if i:
                    f_sum += i[0]
                    t_sum += i[1]
                if m:
                    m_sum += m[1]
        return f_sum, t_sum, m_sum

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

# ==========================================
# 個股詳細圖表顯示函式 (強效修正 DataFrame 結構)
# ==========================================
@st.fragment
def render_tactical_guide(stock_id, stock_name, price, s_ma_val, l_ma_val, t_sum5, f_sum5, m_sum5, pct_change, short_ma):
    st.divider()
    st.subheader(f"💡 {stock_name} 專屬持股戰術指南")
    
    # 1. 計算這檔股票最新的均線狀態
    ma_diff = abs(s_ma_val - l_ma_val) / l_ma_val
    if ma_diff < 0.025: # 短長均線差距 2.5% 以內視為糾結
        ma_state = "糾結"
    elif (price > s_ma_val) and (s_ma_val > l_ma_val):
        ma_state = "多頭"
    else:
        ma_state = "空頭"
        
    # 2. 依據真實籌碼數據判定訊號
    if t_sum5 > 0 and f_sum5 > 0:
        signal = "🚀 土洋雙加碼"
    elif t_sum5 > 0:
        signal = "🔥 投信鎖碼"
    elif f_sum5 > 500:
        signal = "💰 外資大掃貨"
    elif m_sum5 > 0 and (f_sum5 < 0 or t_sum5 < 0):
        signal = "💀 散戶接刀"
    else:
        if pct_change >= 2:
            signal = "🔥 買盤進駐"
        elif pct_change <= -2:
            signal = "💀 賣壓沉重"
        else:
            signal = "🟢 籌碼整理"

    # 3. 顯示目前偵測到的狀態
    st.markdown(f"**目前偵測型態：** 均線 `{ma_state}` ｜ 籌碼 `{signal}`")
    
    # 4. 讓使用者點選持股情境
    user_status = st.radio(
        "請選擇您目前對本檔股票的狀態：",
        ["🛒 未購入 (評估進場)", "📈 已持股 (帳面獲利中)", "📉 已持股 (帳面套牢中)"],
        horizontal=True,
        key=f"status_{stock_id}"
    )
    
    # 5. 依據選擇與股票狀態給予專屬提示
    action_tip, desc = "", ""
    if user_status == "🛒 未購入 (評估進場)":
        if signal in ["🚀 土洋雙加碼", "🔥 投信鎖碼", "💰 外資大掃貨", "🔥 買盤進駐"]:
            if ma_state == "多頭":
                action_tip, desc = "🔥 突破追漲", "極強順勢型態，適合分批追價，沿短均線持有。"
            elif ma_state == "糾結":
                action_tip, desc = "👀 潛伏卡位", "爆發前夕卡位點，盈虧比極佳，帶量突破可加碼。"
            else:
                action_tip, desc = "🌱 低檔試探", "左側交易抄底，建議小資金試水溫，嚴守前低停損。"
        elif "接刀" in signal or "賣壓" in signal:
            action_tip, desc = "🚨 觀望避開", "籌碼極差，縱使下跌也不可入場接刀。"
        else:
            if ma_state == "多頭":
                action_tip, desc = "📈 技術偏多", "籌碼雖不明顯，但技術面強勢，可拉回短均買進。"
            else:
                action_tip, desc = "⏳ 觀望待變", "籌碼與技術面皆無明顯起漲訊號，建議先觀望。"

    elif user_status == "📈 已持股 (帳面獲利中)":
        if signal in ["🚀 土洋雙加碼", "🔥 投信鎖碼", "💰 外資大掃貨", "🔥 買盤進駐"]:
            if price >= s_ma_val:
                action_tip, desc = "🔥 順勢加碼 / 續抱", f"主力持續偏多，股價在短均線 ({short_ma}日) 之上，可讓獲利奔跑或逢回支撐加碼。"
            else:
                action_tip, desc = "⚠️ 跌破短均", f"籌碼雖佳但跌破短均線 ({short_ma}日)，建議部分獲利了結。"
        elif "接刀" in signal or "賣壓" in signal:
            action_tip, desc = "⚠️ 逢高減碼", "主力有逢高出貨跡象，建議分批落袋為安。"
        else:
            action_tip, desc = "🚀 獲利續抱", "趨勢未變，沿短均線持股，不預設高點。"

    else: # 📉 已持股 (帳面套牢中)
        if signal in ["🚀 土洋雙加碼", "🔥 投信鎖碼", "💰 外資大掃貨"]:
            action_tip, desc = "🌱 試探性攤平", "法人有逆勢護盤跡象，可考慮小額攤平一次，嚴守前低停損。"
        elif "接刀" in signal or "賣壓" in signal:
            action_tip, desc = "🚨 嚴禁攤平 / 停損減碼", "主力全面倒貨且趨勢破位，絕不能攤平，建議果斷停損。"
        else:
            action_tip, desc = "⏳ 觀望勿攤平", "未見止跌訊號，越攤平只會套越深，靜待轉折。"

    # 6. 顯示最終結論
    st.info(f"### {action_tip}\n{desc}")



def show_single_stock_detail(stock_id):
    custom_name = st.query_params.get("name")
    if custom_name:
        stock_name = custom_name
    elif stock_id in CN_NAME_MAP:
        stock_name = f"{stock_id} {CN_NAME_MAP[stock_id]}"
    else:
        stock_name = stock_id

    st.subheader(f"📊 股票代號：{stock_name} 詳細技術與籌碼分析")
    
    # 下載歷史資料
    df = yf.download(f"{stock_id}.TW", period="6mo", auto_adjust=True)
    if df.empty:
        st.error(f"❌ 查無代號 {stock_id} 的行情資料，請確認股號是否正確。")
        return

    # 【關鍵修復】打平 yfinance 傳回來的 MultiIndex 雙層欄位
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 確保數值轉為純 1D 系列
    df['Open'] = pd.to_numeric(df['Open'].squeeze(), errors='coerce')
    df['High'] = pd.to_numeric(df['High'].squeeze(), errors='coerce')
    df['Low'] = pd.to_numeric(df['Low'].squeeze(), errors='coerce')
    df['Close'] = pd.to_numeric(df['Close'].squeeze(), errors='coerce')
    df['Volume'] = pd.to_numeric(df['Volume'].squeeze(), errors='coerce')

    custom_cfg = CUSTOM_MA_DB.get(stock_id, {})
    short_ma = custom_cfg.get('short') or find_best_ma_v2(df, 16, 25)
    long_ma = custom_cfg.get('long') or find_best_ma_v2(df, 45, 70)
    df['MS'] = df['Close'].rolling(window=short_ma).mean()
    df['ML'] = df['Close'].rolling(window=long_ma).mean()

    crawler = ChipCrawlerV160(stock_id)
    recent_dates = [d.to_pydatetime().date() for d in df.index[-10:]]
    target_date = recent_dates[-1]
    
    m, i, s = crawler.get_latest_chip_summary(target_date)
    f_sum5, t_sum5, m_sum5 = crawler.get_multi_day_summary(recent_dates, lookback_days=5)

    price = float(df['Close'].iloc[-1])
    prev_price = float(df['Close'].iloc[-2]) if len(df) > 1 else price
    change = price - prev_price
    pct_change = (change / prev_price * 100) if prev_price != 0 else 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric("現價", f"{price:.2f}")
    c2.metric("漲跌", f"{change:+.2f}")
    c3.metric("漲跌幅", f"{pct_change:+.2f}%")

    st.markdown("### 🔍 近 5 日籌碼累積詳情")
    cols = st.columns(3)
    with cols[0]:
        st.metric("近 5 日外資累計", f"{f_sum5:+d} 張", delta=f"當日: {i[0]:+d} 張" if i else None)
    with cols[1]:
        st.metric("近 5 日投信累計", f"{t_sum5:+d} 張", delta=f"當日: {i[1]:+d} 張" if i else None)
    with cols[2]:
        st.metric("近 5 日融資累計", f"{m_sum5:+d} 張", delta=f"當日: {m[1]:+d} 張" if m else None)

    # 取最近 120 根 K 線畫圖
    p_df = df.tail(120).copy()
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])

    # K 棒繪製
    fig.add_trace(go.Candlestick(
        x=p_df.index,
        open=p_df['Open'],
        high=p_df['High'],
        low=p_df['Low'],
        close=p_df['Close'],
        name='K棒',
        increasing_line_color='#ef5350',
        decreasing_line_color='#26a69a'
    ), row=1, col=1)
    
    # 均線繪製
    fig.add_trace(go.Scatter(
        x=p_df.index, y=p_df['MS'], mode='lines', 
        name=f'短均({short_ma}日)', line=dict(color='#ff9800', width=1.5)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=p_df.index, y=p_df['ML'], mode='lines', 
        name=f'長均({long_ma}日)', line=dict(color='#9c27b0', width=1.5)
    ), row=1, col=1)
    
    # 成交量柱狀圖繪製
    v_cols = ['#ef5350' if c >= o else '#26a69a' for c, o in zip(p_df['Close'], p_df['Open'])]
    fig.add_trace(go.Bar(
        x=p_df.index, y=p_df['Volume'], 
        name='成交量', marker_color=v_cols
    ), row=2, col=1)
    
    fig.update_layout(
        height=550,
        template="plotly_white",
        xaxis_rangeslider_visible=False,
        showlegend=True,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified"
    )
    fig.update_yaxes(side="right")
    
    st.plotly_chart(fig, use_container_width=True)

    # =================================================================
    # ⭐ 新增：個股專屬戰術指南 (動態分析本檔股票)
    # =================================================================
    st.divider()
    st.subheader(f"💡 {stock_name} 專屬持股戰術指南")
    
    # 1. 計算這檔股票最新的均線狀態
    s_ma_val = p_df['MS'].iloc[-1]
    l_ma_val = p_df['ML'].iloc[-1]
    
    ma_diff = abs(s_ma_val - l_ma_val) / l_ma_val
    if ma_diff < 0.025: # 短長均線差距 2.5% 以內視為糾結
        ma_state = "糾結"
    elif (price > s_ma_val) and (s_ma_val > l_ma_val):
        ma_state = "多頭"
    else:
        ma_state = "空頭"
        
    # 2. 依據真實籌碼數據判定訊號
    if t_sum5 > 0 and f_sum5 > 0:
        signal = "🚀 土洋雙加碼"
    elif t_sum5 > 0:
        signal = "🔥 投信鎖碼"
    elif f_sum5 > 500:
        signal = "💰 外資大掃貨"
    elif m_sum5 > 0 and (f_sum5 < 0 or t_sum5 < 0):
        signal = "💀 散戶接刀"
    else:
        if pct_change >= 2:
            signal = "🔥 買盤進駐"
        elif pct_change <= -2:
            signal = "💀 賣壓沉重"
        else:
            signal = "🟢 籌碼整理"

    # 3. 顯示目前偵測到的狀態
    st.markdown(f"**目前偵測型態：** 均線 `{ma_state}` ｜ 籌碼 `{signal}`")
    
    # 4. 讓使用者點選持股情境
    user_status = st.radio(
        "請選擇您目前對本檔股票的狀態：",
        ["🛒 未購入 (評估進場)", "📈 已持股 (帳面獲利中)", "📉 已持股 (帳面套牢中)"],
        horizontal=True,
        key=f"status_{stock_id}"
    )
    
    # 5. 依據選擇與股票狀態給予專屬提示
    action_tip = ""
    desc = ""
    
    if user_status == "🛒 未購入 (評估進場)":
        if signal in ["🚀 土洋雙加碼", "🔥 投信鎖碼", "💰 外資大掃貨", "🔥 買盤進駐"]:
            if ma_state == "多頭":
                action_tip, desc = "🔥 突破追漲", "極強順勢型態，適合分批追價，沿短均線持有。"
            elif ma_state == "糾結":
                action_tip, desc = "👀 潛伏卡位", "爆發前夕卡位點，盈虧比極佳，帶量突破可加碼。"
            else:
                action_tip, desc = "🌱 低檔試探", "左側交易抄底，建議小資金試水溫，嚴守前低停損。"
        elif "接刀" in signal or "賣壓" in signal:
            action_tip, desc = "🚨 觀望避開", "籌碼極差，縱使下跌也不可入場接刀。"
        else:
            if ma_state == "多頭":
                action_tip, desc = "📈 技術偏多", "籌碼雖不明顯，但技術面強勢，可拉回短均買進。"
            else:
                action_tip, desc = "⏳ 觀望待變", "籌碼與技術面皆無明顯起漲訊號，建議先觀望。"

    elif user_status == "📈 已持股 (帳面獲利中)":
        if signal in ["🚀 土洋雙加碼", "🔥 投信鎖碼", "💰 外資大掃貨", "🔥 買盤進駐"]:
            if price >= s_ma_val:
                action_tip, desc = "🔥 順勢加碼 / 續抱", f"主力持續偏多，股價在短均線 ({short_ma}日) 之上，可讓獲利奔跑或逢回支撐加碼。"
            else:
                action_tip, desc = "⚠️ 跌破短均", f"籌碼雖佳但跌破短均線 ({short_ma}日)，建議部分獲利了結。"
        elif "接刀" in signal or "賣壓" in signal:
            action_tip, desc = "⚠️ 逢高減碼", "主力有逢高出貨跡象，建議分批落袋為安。"
        else:
            action_tip, desc = "🚀 獲利續抱", "趨勢未變，沿短均線持股，不預設高點。"

    else: # 📉 已持股 (帳面套牢中)
        if signal in ["🚀 土洋雙加碼", "🔥 投信鎖碼", "💰 外資大掃貨"]:
            action_tip, desc = "🌱 試探性攤平", "法人有逆勢護盤跡象，可考慮小額攤平一次，嚴守前低停損。"
        elif "接刀" in signal or "賣壓" in signal:
            action_tip, desc = "🚨 嚴禁攤平 / 停損減碼", "主力全面倒貨且趨勢破位，絕不能攤平，建議果斷停損。"
        else:
            action_tip, desc = "⏳ 觀望勿攤平", "未見止跌訊號，越攤平只會套越深，靜待轉折。"

    # 6. 顯示最終結論
    st.info(f"### {action_tip}\n{desc}")

# ==========================================
# Streamlit 主介面
# ==========================================
show_stock_id = st.query_params.get("stock")
if show_stock_id:
    show_single_stock_detail(show_stock_id)
    st.stop()

st.title("📡 台股強勢股快篩 (V170 戰略解讀版)")
tab_main, tab_guide = st.tabs(["🎯 籌碼掃描主頁", "💡 持股戰術指南"])
    
with tab_main:
    
    
    option = st.radio(
        "選擇掃描模式", 
        ["自選股票", "熱門板塊指標股 (50檔龍頭)", "熱門飆股動態快篩 (當日成交量Top50)"],
        index=0
    )
    
    stock_list = []
    if option == "自選股票":
        stock_input = st.text_input("輸入股票代號 (空白隔開)", "2330 2454 2603 2313")
        stock_list = stock_input.split()
    elif option == "熱門板塊指標股 (50檔龍頭)":
        all_codes = []
        for sector in SECTOR_DB.values():
            all_codes += sector
        stock_list = list(set(all_codes))
    else:
        top_tickers = [
            "2330", "2317", "2454", "2382", "3231", 
            "2603", "2609", "2615", "1519", "1504",
            "2356", "2376", "3037", "2303", "3711",
            "6669", "3661", "3443", "8046", "3035"
        ]
        stock_list = top_tickers
    
    results = []
    if st.button("開始掃描"):
        tickers = [f"{c}.TW" for c in stock_list]
        try:
            data = yf.download(" ".join(tickers), period="3mo", group_by='ticker', auto_adjust=True)
        except Exception as e:
            st.error("❌ 無法下載行情資料，請檢查網路。")
            st.stop()
    
        for stock_id in stock_list:
            try:
                df_s = data[f"{stock_id}.TW"].copy() if f"{stock_id}.TW" in data else pd.DataFrame()
                if isinstance(df_s.columns, pd.MultiIndex):
                    df_s.columns = df_s.columns.get_level_values(0)
                    
                if df_s.empty or df_s['Close'].dropna().empty:
                    continue
                df_s = df_s.dropna(subset=['Close'])
                
                last_p = float(df_s['Close'].iloc[-1])
                prev_p = float(df_s['Close'].iloc[-2])
                pct_chg = ((last_p - prev_p) / prev_p) * 100
    
                if stock_id in CN_NAME_MAP:
                    s_name = CN_NAME_MAP[stock_id]
                else:
                    s_name = ""
    
                display_name = f"{stock_id} {s_name}".strip()
    
                crawler = ChipCrawlerV160(stock_id)
                recent_dates = [d.to_pydatetime().date() for d in df_s.index[-10:]]
                f_sum5, t_sum5, m_sum5 = crawler.get_multi_day_summary(recent_dates, lookback_days=5)
    
                if t_sum5 >= 300 and m_sum5 <= 0:
                    signal = "🔥 投信鎖碼 (法人吃貨/散戶退場)"
                elif f_sum5 >= 1000 and t_sum5 >= 300:
                    signal = "🚀 土洋雙加碼 (主力強勢作多)"
                elif f_sum5 >= 1500:
                    signal = "💰 外資大掃貨 (外資波段佈局)"
                elif t_sum5 >= 500:
                    signal = "🎯 投信急買 (波段作多訊號)"
                elif f_sum5 < -1000 and m_sum5 > 500:
                    signal = "💀 散戶接刀 (法人倒貨/融資暴增)"
                else:
                    signal = "🟢 籌碼整理中"
    
                results.append({
                    "標的": display_name,
                    "收盤價": f"{last_p:.2f}",
                    "漲跌幅": f"{pct_chg:+.2f}%",
                    "籌碼戰略解讀": signal,
                    "5日外資": f"{f_sum5:+d} 張",
                    "5日投信": f"{t_sum5:+d} 張",
                    "5日融資": f"{m_sum5:+d} 張",
                    "raw_id": stock_id
                })
            except Exception as e:
                continue
    
        if results:
            res_df = pd.DataFrame(results)
            res_df["查看分頁"] = res_df.apply(lambda row: f"?stock={row['raw_id']}&name={row['標的']}", axis=1)
            res_df = res_df.drop(columns=["raw_id"])
            
            st.dataframe(
                res_df,
                column_config={
                    "查看分頁": st.column_config.LinkColumn(
                        "詳細圖表",
                        display_text="📊 開啟 K 線圖"
                    )
                },
                use_container_width=True,
                hide_index=True
            )
        else:
            st.warning("⚠️ 查無符合條件的股票")

with tab_guide:
    st.header("💡 持股戰術提示與情境決策指南")
    st.caption("請根據您目前的「持股狀態」切換檢視專屬戰術指南：")
    
    user_status = st.radio(
        "選擇您的個人持股情境：",
        ["🛒 未購入 (找進場點)", "📈 獲利持股中 (找加碼/停利點)", "📉 虧損套牢中 (評估攤平/停損)"],
        horizontal=True
    )
    
# 呼叫局部更新戰術指南區塊
    render_tactical_guide(
        stock_id=stock_id,
        stock_name=stock_name,
        price=price,
        s_ma_val=p_df['MS'].iloc[-1],
        l_ma_val=p_df['ML'].iloc[-1],
        t_sum5=t_sum5,
        f_sum5=f_sum5,
        m_sum5=m_sum5,
        pct_change=pct_change,
        short_ma=short_ma
    )
