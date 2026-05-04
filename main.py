import os
import json
import datetime
import traceback
import re
import concurrent.futures
from flask import Flask, request, jsonify
from flask_cors import CORS 
import yfinance as yf
import pandas as pd
import numpy as np
import google.generativeai as genai
from FinMind.data import DataLoader

app = Flask(__name__)
CORS(app) 

api_key = os.environ.get("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

STOCK_DICT = {
    "2382": "廣達", "3231": "緯創", "2376": "技嘉", "3324": "雙鴻", "3017": "奇鋐",
    "3661": "世芯-KY", "3131": "弘塑", "6187": "萬潤", "6683": "雍智科技", "3583": "辛耘",
    "3163": "波若威", "3363": "上詮", "4979": "華星光", "6442": "光聖", "4908": "前鼎",
    "2504": "國產", "2515": "中工", "2520": "冠德", "1436": "華友聯", "2501": "國建",
    "1503": "士電", "1504": "東元", "1513": "中興電", "1514": "亞力", "1519": "華城",
    "2330": "台積電", "2317": "鴻海", "2454": "聯發科", "2301": "光寶科"
}

SECTORS = {
    "🔥 AI 伺服器 & 散熱": ["2382.TW", "3231.TW", "2376.TW", "3324.TW", "3017.TW"],
    "🚀 CoWoS 先進封裝": ["3661.TW", "3131.TW", "6187.TW", "6683.TW", "3583.TW"],
    "⚡ CPO 矽光子通訊": ["3163.TW", "3363.TW", "4979.TW", "6442.TW", "4908.TW"],
    "🏗️ 營造建材 (內需)": ["2504.TW", "2515.TW", "2520.TW", "1436.TW", "2501.TW"],
    "🔋 重電與綠能": ["1503.TW", "1504.TW", "1513.TW", "1514.TW", "1519.TW"]
}

RADAR_WATCHLIST = [s for group in SECTORS.values() for s in group] + ['2330.TW', '2317.TW']

@app.route('/')
def home():
    return "AI 戰情室大腦運轉中！(搭載最高階操盤手 Prompt 與嚴謹邏輯模塊)"

def fetch_stock_basic(symbol):
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period="5d", interval="1d")
        if df.empty and symbol.endswith('.TW'):
            symbol = symbol.replace('.TW', '.TWO')
            stock = yf.Ticker(symbol)
            df = stock.history(period="5d", interval="1d")
        
        if len(df) >= 2:
            curr = df.iloc[-1]['Close']
            prev = df.iloc[-2]['Close']
            change_pct = ((curr - prev) / prev) * 100
            pure_sym = symbol.split('.')[0]
            name = STOCK_DICT.get(pure_sym, stock.info.get('shortName', pure_sym))
            return {"symbol": pure_sym, "name": name, "price": round(curr, 2), "change": round(change_pct, 2)}
    except: pass
    return None

@app.route('/sectors', methods=['GET'])
def get_sectors():
    sector_results = {}
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            for sector_name, symbols in SECTORS.items():
                results = list(executor.map(fetch_stock_basic, symbols))
                valid_results = [r for r in results if r]
                valid_results.sort(key=lambda x: x['change'], reverse=True)
                sector_results[sector_name] = valid_results
        return jsonify({"status": "success", "data": sector_results})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

def check_radar_symbol(symbol):
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period="1mo", interval="1d")
        if df.empty and symbol.endswith('.TW'):
            symbol = symbol.replace('.TW', '.TWO')
            stock = yf.Ticker(symbol)
            df = stock.history(period="1mo", interval="1d")
        if df.empty or len(df) < 26: return None

        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD_Signal'] = (df['EMA12'] - df['EMA26']).ewm(span=9, adjust=False).mean()
        df['OSC'] = (df['EMA12'] - df['EMA26']) - df['MACD_Signal']
        df['9_high'] = df['High'].rolling(9).max()
        df['9_low'] = df['Low'].rolling(9).min()
        df['RSV'] = ((df['Close'] - df['9_low']) / (df['9_high'] - df['9_low']) * 100).fillna(50)
        
        K, D = [], []
        pk, pd_val = 50, 50
        for rsv in df['RSV'].tolist():
            ck = (2/3)*pk + (1/3)*rsv; cd = (2/3)*pd_val + (1/3)*ck
            K.append(ck); D.append(cd); pk, pd_val = ck, cd
        df['K'], df['D'] = K, D

        last_2 = df.tail(2)
        kd_cross = (last_2.iloc[0]['K'] <= last_2.iloc[0]['D']) and (last_2.iloc[1]['K'] > last_2.iloc[1]['D'])
        trend_up = (last_2.iloc[1]['Close'] > last_2.iloc[1]['MA20']) and (last_2.iloc[1]['OSC'] > 0)

        if kd_cross and trend_up:
            pure_sym = symbol.split('.')[0]
            name = STOCK_DICT.get(pure_sym, stock.info.get('shortName', pure_sym))
            return {"symbol": pure_sym, "name": name, "price": round(last_2.iloc[1]['Close'], 2)}
    except: pass
    return None

@app.route('/radar', methods=['GET'])
def radar():
    matched = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            for r in executor.map(check_radar_symbol, RADAR_WATCHLIST):
                if r: matched.append(r)
        return jsonify({"status": "success", "matches": matched})
    except Exception as e: return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/predict', methods=['GET'])
def predict():
    symbol = request.args.get('symbol', '2330.TW')
    interval = request.args.get('interval', '1d')
    if interval not in ['1m', '5m', '15m', '60m', '1d']: interval = '1d'

    try:
        period = "5d" if interval in ['1m', '5m'] else ("1mo" if interval in ['15m', '60m'] else "6mo")
        stock = yf.Ticker(symbol)
        df = stock.history(period=period, interval=interval)
        
        if df.empty and symbol.endswith('.TW'):
            fallback_symbol = symbol.replace('.TW', '.TWO')
            stock = yf.Ticker(fallback_symbol)
            df = stock.history(period=period, interval=interval)
            symbol = fallback_symbol

        if df.empty: return jsonify({"status": "error", "message": "查無資料，請確認代碼"}), 400

        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA10'] = df['Close'].rolling(window=10).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA60'] = df['Close'].rolling(window=60).mean()
        df['BB_std'] = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['MA20'] + 2 * df['BB_std']
        df['BB_lower'] = df['MA20'] - 2 * df['BB_std']
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['DIF'] = df['EMA12'] - df['EMA26']
        df['MACD_Signal'] = df['DIF'].ewm(span=9, adjust=False).mean()
        df['OSC'] = df['DIF'] - df['MACD_Signal']
        df['9_high'] = df['High'].rolling(9).max()
        df['9_low'] = df['Low'].rolling(9).min()
        df['RSV'] = ((df['Close'] - df['9_low']) / (df['9_high'] - df['9_low']) * 100).fillna(50)
        
        K, D = [], []
        pk, pd_val = 50, 50
        for rsv in df['RSV'].tolist():
            ck = (2/3)*pk + (1/3)*rsv; cd = (2/3)*pd_val + (1/3)*ck
            K.append(ck); D.append(cd); pk, pd_val = ck, cd
        df['K'], df['D'] = K, D
        df['Volume_Dir'] = np.sign(df['Close'].diff()).fillna(0)
        df['OBV'] = (df['Volume'] * df['Volume_Dir']).cumsum()

        df = df.fillna(0)
        chart_data, macd_data, kd_data, obv_data = [], [], [], []
        for date, row in df.tail(80).iterrows():
            tv = date.strftime('%Y-%m-%d') if interval == '1d' else int(date.timestamp())
            chart_data.append({
                "time": tv, "open": round(row['Open'],2), "high": round(row['High'],2), "low": round(row['Low'],2), "close": round(row['Close'],2), 
                "ma5": row['MA5'], "ma20": row['MA20'], "ma60": row['MA60'], "bb_upper": row['BB_upper'], "bb_lower": row['BB_lower']
            })
            macd_data.append({"time": tv, "dif": row['DIF'], "signal": row['MACD_Signal'], "osc": row['OSC']})
            kd_data.append({"time": tv, "k": row['K'], "d": row['D']})
            obv_data.append({"time": tv, "value": row['OBV']})

        current_price = round(float(df['Close'].iloc[-1]), 2)
        pure_symbol = symbol.split('.')[0]
        
        fun_data = {"eps": "--", "pe_ratio": "--"}
        try:
            info = stock.info
            display_name = STOCK_DICT.get(pure_symbol, info.get('shortName', pure_symbol))
            eps, pe = info.get("trailingEps"), info.get("trailingPE")
            if eps: fun_data["eps"] = round(eps, 2)
            if pe: fun_data["pe_ratio"] = round(pe, 2)
        except: display_name = STOCK_DICT.get(pure_symbol, pure_symbol)

        chip_info, chip_chart_data, chip_table_data, foreign_data, trust_data = "無近期資料", [], [], [], []
        if interval == '1d':
            try:
                dl = DataLoader()
                df_chips = dl.taiwan_stock_institutional_investors(stock_id=pure_symbol, start_date=(datetime.datetime.now() - datetime.timedelta(days=30)).strftime('%Y-%m-%d'))
                if not df_chips.empty:
                    df_chips['net_buy'] = df_chips['buy'] - df_chips['sell']
                    df_chips['name'] = df_chips['name'].replace({'外資及陸資(不含外資自營商)': '外資', '外資及陸資': '外資', '自營商(自行買賣)': '自營', '自營商(避險)': '自營', '自營商': '自營'})
                    pv = df_chips.groupby(['date', 'name'])['net_buy'].sum().unstack(fill_value=0).reset_index()
                    for col in ['外資', '投信', '自營']:
                        if col not in pv.columns: pv[col] = 0
                    pv['合計'] = pv['外資'] + pv['投信'] + pv['自營']
                    for _, r in pv.iterrows():
                        t_str = str(r['date'])
                        chip_chart_data.append({"time": t_str, "value": round(r['合計']/1000, 2)})
                        foreign_data.append({"time": t_str, "value": round(r['外資']/1000, 2)})
                        trust_data.append({"time": t_str, "value": round(r['投信']/1000, 2)})
                    
                    # 提速：給 AI 看近 5 日籌碼即可判斷意圖
                    chip_info = pv.tail(5).to_string() 
                    for _, r in pv.tail(10).iloc[::-1].iterrows():
                        chip_table_data.append({"date": str(r['date'])[5:], "foreign": round(r['外資']/1000,1), "trust": round(r['投信']/1000,1), "dealer": round(r['自營']/1000,1), "total": round(r['合計']/1000,1)})
            except: pass

        tech_str = df[['Close', 'MA20', 'OSC', 'K', 'D']].tail(3).to_string()
        last_row = df.iloc[-1]

        # === 【最高階量化操盤手提示詞 (God-Tier Prompt)】 ===
        # 嚴格約束 AI 的思考路徑，禁止腦補，要求精準解讀主力意圖
        prompt = (
            f"你是一位具備 20 年實戰經驗的台股頂級量化操盤手，專精於「價格行為」、「籌碼微結構」與「風險報酬比」精算。\n"
            f"請分析 {display_name} ({pure_symbol}) 的近期數據。\n\n"
            f"【核心判讀規則】：\n"
            f"1. 趨勢與型態：依據技術面數據判定。若無明確的雙底(W)或雙頂(M)，必須客觀回報「未形成標準型態」或「區間震盪」，嚴禁過度腦補。\n"
            f"2. 籌碼解構：檢視籌碼面，不可只報數字，必須解讀背後意圖（如：外資連續佈局、投信逢高結帳、土洋對作）。\n"
            f"3. 機率評估：依據技術動能與籌碼流向互抵結果，給出明日上漲、下跌、震盪的具體機率 (總和必須精準等於 100)。\n"
            f"4. 實戰劇本：明確訂出具備防守邏輯的「突破買點」、「區間操作點」、「嚴格停損點」。\n\n"
            f"務必只輸出純 JSON，格式如下 (絕對不可包含 Markdown 或額外文字)：\n"
            f"{{\n"
            f"  \"signal\": \"偏多/偏空/震盪\", \"pressure\": \"壓力價\", \"support\": \"支撐價\", \"stop_loss\": \"停損參考價\",\n"
            f"  \"prob_up\": 45, \"prob_down\": 25, \"prob_flat\": 30,\n"
            f"  \"pattern_kline\": \"嚴謹型態判定(10字內)\", \"pattern_trend\": \"均線與布林狀態(10字內)\",\n"
            f"  \"chip_status\": \"法人主力意圖精簡結論(15字內)\",\n"
            f"  \"industry_desc\": \"所屬產業模塊(10字內)\", \"related_stocks\": \"代表性連動概念股\",\n"
            f"  \"scenario_up\": {{\"price\": \"突破價\", \"action\": \"向上突破策略\"}},\n"
            f"  \"scenario_flat\": {{\"price\": \"區間價\", \"action\": \"震盪盤整策略\"}},\n"
            f"  \"scenario_down\": {{\"price\": \"跌破價\", \"action\": \"防守停損策略\"}},\n"
            f"  \"stars\": 3, \"advice\": [\"趨勢與籌碼深度綜合總結1\", \"操作風險管控提示2\"]\n"
            f"}}\n\n"
            f"基本面：{fun_data}\n技術面：{tech_str}\n籌碼面：{chip_info}"
        )
        
        # 本地端備援大腦
        fallback_signal = "區間震盪"
        if last_row['K'] > last_row['D'] and last_row['Close'] > last_row['MA20']: fallback_signal = "多頭格局"
        elif last_row['K'] < last_row['D'] and last_row['Close'] < last_row['MA20']: fallback_signal = "空頭弱勢"
        
        ai_data = {
            "signal": fallback_signal, 
            "pressure": str(round(last_row['BB_upper'], 2)), 
            "support": str(round(last_row['BB_lower'], 2)), 
            "stop_loss": str(round(last_row['MA20'], 2)), 
            "prob_up": 45 if fallback_signal == "多頭格局" else 25, 
            "prob_down": 25 if fallback_signal == "多頭格局" else 45, 
            "prob_flat": 30,
            "pattern_kline": "量化模型計算中", "pattern_trend": "均線與布林計算中", 
            "chip_status": "請參考左方三大法人明細", # 新增籌碼狀態欄位
            "industry_desc": "台股上市櫃", "related_stocks": "同族群個股",
            "scenario_up": {"price": str(round(last_row['BB_upper'], 2)), "action": "突破上軌順勢偏多"}, 
            "scenario_flat": {"price": str(round(last_row['Close'], 2)), "action": "均線附近來回操作"}, 
            "scenario_down": {"price": str(round(last_row['MA20'], 2)), "action": "跌破月線嚴格停損"},
            "stars": 3, "advice": ["AI 雲端連線逾時，已自動切換本地量化備援大腦", "此為純技術指標運算結果，請親自搭配籌碼流向判斷"]
        }
        
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0.1))
            text = response.text
            match = re.search(r'\{[\s\S]*\}', text)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                    for k, v in parsed.items():
                        if k in ai_data: ai_data[k] = v
                except:
                    # JSON 解析失敗的暴力 Regex 挖礦 (包含新增的 chip_status)
                    t = match.group(0)
                    def ext_str(key, default): 
                        m = re.search(f'"{key}"\s*:\s*"([^"]+)"', t)
                        return m.group(1) if m else default
                    def ext_int(key, default):
                        m = re.search(f'"{key}"\s*:\s*(\d+)', t)
                        return int(m.group(1)) if m else default

                    ai_data["signal"] = ext_str("signal", ai_data["signal"])
                    ai_data["pressure"] = ext_str("pressure", ai_data["pressure"])
                    ai_data["support"] = ext_str("support", ai_data["support"])
                    ai_data["stop_loss"] = ext_str("stop_loss", ai_data["stop_loss"])
                    ai_data["pattern_kline"] = ext_str("pattern_kline", "未形成標準型態")
                    ai_data["pattern_trend"] = ext_str("pattern_trend", "均線整理")
                    ai_data["chip_status"] = ext_str("chip_status", "法人動向不明")
                    ai_data["industry_desc"] = ext_str("industry_desc", "台股重點板塊")
                    ai_data["related_stocks"] = ext_str("related_stocks", "--")
                    
                    pu = ext_int("prob_up", 33); pd_ = ext_int("prob_down", 33); pf = ext_int("prob_flat", 34)
                    if pu > 0 or pd_ > 0 or pf > 0:
                        ai_data["prob_up"]=pu; ai_data["prob_down"]=pd_; ai_data["prob_flat"]=pf

                    for sc in ["scenario_up", "scenario_flat", "scenario_down"]:
                        m_sc = re.search(f'"{sc}"\s*:\s*{{([^}}]+)}}', t)
                        if m_sc:
                            in_text = m_sc.group(1)
                            p_match = re.search(r'"price"\s*:\s*"([^"]+)"', in_text)
                            a_match = re.search(r'"action"\s*:\s*"([^"]+)"', in_text)
                            if p_match: ai_data[sc]["price"] = p_match.group(1)
                            if a_match: ai_data[sc]["action"] = a_match.group(1)
                    ai_data["advice"] = ["AI 系統已透過暴力演算法萃取部分資料", "建議搭配圖表自行確認"]
        except Exception as e: print("AI 處理異常，啟用備援", e)

        return jsonify({
            "status": "success", "symbol": symbol, "current_price": current_price, "interval": interval,
            "chart_data": chart_data, "macd_data": macd_data, "kd_data": kd_data, 
            "obv_data": obv_data, "chip_data": chip_chart_data, 
            "foreign_data": foreign_data, "trust_data": trust_data,
            "chip_table": chip_table_data, "fundamental": fun_data, "ai_analysis": ai_data
        })
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"status": "error", "message": f"伺服器錯誤: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))
