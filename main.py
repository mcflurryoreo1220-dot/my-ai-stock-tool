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

RADAR_WATCHLIST = ['2330.TW', '2317.TW', '2454.TW', '2382.TW', '3231.TW', '2603.TW', '1519.TW', '3661.TW', '6285.TW', '6147.TWO', '6269.TW', '2441.TW']

@app.route('/')
def home():
    return "AI 戰情室大腦運轉中！(搭載高階劇本與產業連動分析)"

def check_radar_symbol(symbol):
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period="1mo", interval="1d")
        if df.empty and symbol.endswith('.TW'):
            fallback = symbol.replace('.TW', '.TWO')
            stock = yf.Ticker(fallback)
            df = stock.history(period="1mo", interval="1d")
            symbol = fallback
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
            return {"symbol": pure_sym, "name": stock.info.get('shortName', pure_sym), "price": round(last_2.iloc[1]['Close'], 2)}
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
            chart_data.append({"time": tv, "open": round(row['Open'],2), "high": round(row['High'],2), "low": round(row['Low'],2), "close": round(row['Close'],2), "ma5": row['MA5'], "ma20": row['MA20'], "ma60": row['MA60']})
            macd_data.append({"time": tv, "dif": row['DIF'], "signal": row['MACD_Signal'], "osc": row['OSC']})
            kd_data.append({"time": tv, "k": row['K'], "d": row['D']})
            obv_data.append({"time": tv, "value": row['OBV']})

        current_price = round(float(df['Close'].iloc[-1]), 2)
        pure_symbol = symbol.split('.')[0]
        
        fun_data = {"eps": "--", "pe_ratio": "--"}
        try:
            info = stock.info
            display_name = info.get('shortName', pure_symbol)
            eps, pe = info.get("trailingEps"), info.get("trailingPE")
            if eps: fun_data["eps"] = round(eps, 2)
            if pe: fun_data["pe_ratio"] = round(pe, 2)
        except: display_name = pure_symbol

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
                    
                    chip_info = pv.tail(5).to_string() 
                    for _, r in pv.tail(10).iloc[::-1].iterrows():
                        chip_table_data.append({"date": str(r['date'])[5:], "foreign": round(r['外資']/1000,1), "trust": round(r['投信']/1000,1), "dealer": round(r['自營']/1000,1), "total": round(r['合計']/1000,1)})
            except: pass

        prompt = (
            f"你是一位精通全球產業鏈的台股量化操盤手。分析 {display_name} ({pure_symbol})。\n"
            f"請輸出 JSON，不要有任何 Markdown。\n"
            f"{{\n"
            f"  \"signal\": \"偏多/偏空/震盪\",\n  \"pressure\": \"壓力價\",\n  \"support\": \"支撐價\",\n  \"stop_loss\": \"停損價\",\n"
            f"  \"pattern_kline\": \"描述近期K線型態(如:量縮整理/長紅突破/高檔震盪,限10字)\",\n"
            f"  \"pattern_trend\": \"描述均線排列(如:多頭排列/跌破月線,限10字)\",\n"
            f"  \"industry_desc\": \"描述所屬產業模塊(如:半導體設備/蘋概股,限15字)\",\n"
            f"  \"related_stocks\": \"列出3-4檔連動概念股(包含具代表性的美股或台股，如:NVDA輝達, 2330台積電)\",\n"
            f"  \"scenario_up\": {{\"price\": \"突破價\", \"action\": \"突破後的建議操作(限15字)\"}},\n"
            f"  \"scenario_flat\": {{\"price\": \"震盪價\", \"action\": \"盤整時的建議操作(限15字)\"}},\n"
            f"  \"scenario_down\": {{\"price\": \"防守價\", \"action\": \"跌破後的建議操作(限15字)\"}},\n"
            f"  \"stars\": 1到5的整數,\n  \"advice\": [\"總結建議1\", \"總結建議2\"]\n"
            f"}}\n\n"
            f"基本面：{fun_data}\n技術面：{df.tail(10).to_string()}\n籌碼面：{chip_info}"
        )
        
        ai_data = None
        for model_name in ['gemini-1.5-flash', 'gemini-1.5-pro']:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0.2))
                text = re.sub(r'```json\n?', '', response.text).replace('```', '').strip()
                match = re.search(r'\{[\s\S]*\}', text)
                if match:
                    ai_data = json.loads(match.group(0))
                    break
            except Exception as e: print(f"[{model_name}] 解析失敗", e)
                
        if not ai_data:
            ai_data = {"signal": "等待連線", "pressure": "--", "support": "--", "stop_loss": "--", 
                       "pattern_kline": "解析中", "pattern_trend": "解析中", "industry_desc": "解析中", "related_stocks": "解析中",
                       "scenario_up": {"price":"--", "action":"--"}, "scenario_flat": {"price":"--", "action":"--"}, "scenario_down": {"price":"--", "action":"--"},
                       "stars": 0, "advice": ["請稍後重新點擊分析"]}

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
