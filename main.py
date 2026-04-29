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

RADAR_WATCHLIST = [
    '2330.TW', '2317.TW', '2454.TW', '2382.TW', '3231.TW', 
    '2603.TW', '1519.TW', '3661.TW', '6285.TW', '6147.TWO', 
    '6269.TW', '2441.TW', '2362.TW', '3481.TW', '2308.TW'
]

@app.route('/')
def home():
    return "AI 戰情室大腦運轉中！(搭載籌碼X光透視引擎)"

def check_radar_symbol(symbol):
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period="1mo", interval="1d")
        
        if df.empty and symbol.endswith('.TW'):
            fallback = symbol.replace('.TW', '.TWO')
            stock = yf.Ticker(fallback)
            df = stock.history(period="1mo", interval="1d")
            symbol = fallback

        if df.empty or len(df) < 26:
            return None

        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['DIF'] = df['EMA12'] - df['EMA26']
        df['MACD_Signal'] = df['DIF'].ewm(span=9, adjust=False).mean()
        df['OSC'] = df['DIF'] - df['MACD_Signal']
        df['9_high'] = df['High'].rolling(9).max()
        df['9_low'] = df['Low'].rolling(9).min()
        df['RSV'] = (df['Close'] - df['9_low']) / (df['9_high'] - df['9_low']) * 100
        df['RSV'] = df['RSV'].replace([np.inf, -np.inf], np.nan)
        
        rsv_list = df['RSV'].fillna(50).tolist()
        K, D = [], []
        prev_k, prev_d = 50, 50
        for rsv in rsv_list:
            curr_k = (2/3) * prev_k + (1/3) * rsv
            curr_d = (2/3) * prev_d + (1/3) * curr_k
            K.append(curr_k)
            D.append(curr_d)
            prev_k, prev_d = curr_k, curr_d
        df['K'], df['D'] = K, D

        last_2 = df.tail(2)
        prev = last_2.iloc[0]
        curr = last_2.iloc[1]

        kd_cross = (prev['K'] <= prev['D']) and (curr['K'] > curr['D'])
        trend_up = (curr['Close'] > curr['MA20']) and (curr['OSC'] > 0)

        if kd_cross and trend_up:
            name = stock.info.get('shortName', symbol.replace('.TW','').replace('.TWO',''))
            return {
                "symbol": symbol.replace('.TW','').replace('.TWO',''),
                "name": name,
                "price": round(curr['Close'], 2)
            }
    except:
        pass
    return None

@app.route('/radar', methods=['GET'])
def radar():
    matched_stocks = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            results = executor.map(check_radar_symbol, RADAR_WATCHLIST)
            for r in results:
                if r: matched_stocks.append(r)
        return jsonify({"status": "success", "matches": matched_stocks})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/predict', methods=['GET'])
def predict():
    symbol = request.args.get('symbol', '2330.TW')
    interval = request.args.get('interval', '1d')
    
    valid_intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
    if interval not in valid_intervals: interval = '1d'

    try:
        if interval in ['1m', '2m', '5m']: period = "5d"
        elif interval in ['15m', '30m', '60m', '90m', '1h']: period = "1mo"
        else: period = "6mo"

        stock = yf.Ticker(symbol)
        df = stock.history(period=period, interval=interval)
        
        if df.empty and symbol.endswith('.TW'):
            fallback_symbol = symbol.replace('.TW', '.TWO')
            stock = yf.Ticker(fallback_symbol)
            df = stock.history(period=period, interval=interval)
            symbol = fallback_symbol

        if df.empty:
            return jsonify({"status": "error", "message": f"無法獲取 {symbol} 數據。"}), 400

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
        df['RSV'] = (df['Close'] - df['9_low']) / (df['9_high'] - df['9_low']) * 100
        df['RSV'] = df['RSV'].replace([np.inf, -np.inf], np.nan)
        
        rsv_list = df['RSV'].fillna(50).tolist()
        K, D = [], []
        prev_k, prev_d = 50, 50
        for rsv in rsv_list:
            curr_k = (2/3) * prev_k + (1/3) * rsv
            curr_d = (2/3) * prev_d + (1/3) * curr_k
            K.append(curr_k)
            D.append(curr_d)
            prev_k, prev_d = curr_k, curr_d
        df['K'], df['D'] = K, D
        
        df['Volume_Dir'] = np.sign(df['Close'].diff()).fillna(0)
        df['OBV'] = (df['Volume'] * df['Volume_Dir']).cumsum()

        df = df.fillna(0)
        df_chart = df.tail(80)
        
        chart_data, macd_data, kd_data, obv_data = [], [], [], []
        for date, row in df_chart.iterrows():
            time_val = date.strftime('%Y-%m-%d') if interval == '1d' else int(date.timestamp())
            chart_data.append({"time": time_val, "open": round(row['Open'],2), "high": round(row['High'],2), "low": round(row['Low'],2), "close": round(row['Close'],2), "ma5": row['MA5'], "ma10": row['MA10'], "ma20": row['MA20'], "ma60": row['MA60']})
            macd_data.append({"time": time_val, "dif": row['DIF'], "signal": row['MACD_Signal'], "osc": row['OSC']})
            kd_data.append({"time": time_val, "k": row['K'], "d": row['D']})
            obv_data.append({"time": time_val, "value": row['OBV']})

        current_price = round(float(df['Close'].iloc[-1]), 2)
        display_name = symbol

        fundamental_data = {"eps": "--", "pe_ratio": "--"}
        try:
            info = stock.info
            display_name = info.get('shortName', symbol)
            eps = info.get("trailingEps")
            pe = info.get("trailingPE")
            if eps is not None: fundamental_data["eps"] = round(eps, 2)
            if pe is not None: fundamental_data["pe_ratio"] = round(pe, 2)
        except: pass

        # === 【X光透視：分離籌碼陣列】 ===
        pure_symbol = symbol.replace('.TW', '').replace('.TWO', '')
        chip_info, chip_chart_data, chip_table_data = "非日線層級", [], []
        foreign_data, trust_data = [], []
        
        if interval == '1d':
            try:
                dl = DataLoader()
                start_date = (datetime.datetime.now() - datetime.timedelta(days=45)).strftime('%Y-%m-%d')
                df_chips = dl.taiwan_stock_institutional_investors(stock_id=pure_symbol, start_date=start_date)
                if not df_chips.empty:
                    df_chips['net_buy'] = df_chips['buy'] - df_chips['sell']
                    df_chips['name'] = df_chips['name'].replace({'外資及陸資(不含外資自營商)': '外資', '外資及陸資': '外資', '自營商(自行買賣)': '自營', '自營商(避險)': '自營', '自營商': '自營'})
                    pivot_df = df_chips.groupby(['date', 'name'])['net_buy'].sum().unstack(fill_value=0).reset_index()
                    for col in ['外資', '投信', '自營']:
                        if col not in pivot_df.columns: pivot_df[col] = 0
                    pivot_df['合計'] = pivot_df['外資'] + pivot_df['投信'] + pivot_df['自營']
                    pivot_df = pivot_df[pivot_df['合計'] != 0].copy()
                    
                    for _, r in pivot_df.iterrows():
                        time_str = str(r['date'])
                        chip_chart_data.append({"time": time_str, "value": round(float(r['合計']) / 1000, 2)})
                        foreign_data.append({"time": time_str, "value": round(float(r['外資']) / 1000, 2)})
                        trust_data.append({"time": time_val, "value": round(float(r['投信']) / 1000, 2)})
                        # 修正上方寫錯的 time_val，統一為 time_str
                        foreign_data[-1]["time"] = time_str
                        trust_data[-1]["time"] = time_str

                    chip_info = pivot_df.tail(10).to_string() 
                    last_10 = pivot_df.tail(10).iloc[::-1]
                    for _, r in last_10.iterrows():
                        try:
                            chip_table_data.append({"date": str(r['date'])[5:], "foreign": round(float(r.get('外資',0))/1000,1), "trust": round(float(r.get('投信',0))/1000,1), "dealer": round(float(r.get('自營',0))/1000,1), "total": round(float(r.get('合計',0))/1000,1)})
                        except: pass
            except Exception as e:
                print("籌碼解析異常:", e)

        prompt = (
            f"你是台股量化操盤手。分析 {display_name} ({interval})。\n純 JSON 輸出。\n"
            f"{{\"signal\": \"多/空/觀望\", \"pressure\": \"價格\", \"support\": \"價格\", \"stop_loss\": \"價格\", \"path_up\": \"路徑\", \"path_down\": \"路徑\", \"stars\": 1到5整數, \"advice\": [\"技術面\", \"籌碼面\", \"建議\"]}}\n\n"
            f"基本面：{fundamental_data}\n技術面：{df.tail(10).to_string()}\n籌碼面：{chip_info}"
        )
        
        ai_data = None
        models_to_try = ['gemini-1.5-flash', 'gemini-1.5-pro-latest']
        for model_name in models_to_try:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0.1))
                text = response.text.replace("```json\n", "").replace("```", "").strip()
                start = text.find('{')
                end = text.rfind('}')
                if start != -1 and end != -1:
                    ai_data = json.loads(text[start:end+1])
                    break
            except Exception: continue
                
        if not ai_data:
            ai_data = {"signal": "系統繁忙", "pressure": "--", "support": "--", "stop_loss": "--", "path_up": "--", "path_down": "--", "stars": 0, "advice": ["圖表已載入", "請重試"]}

        return jsonify({
            "status": "success", "symbol": symbol, "current_price": current_price, "interval": interval,
            "chart_data": chart_data, "macd_data": macd_data, "kd_data": kd_data, 
            "obv_data": obv_data, "chip_data": chip_chart_data, 
            "foreign_data": foreign_data, "trust_data": trust_data, # 送出拆分的籌碼
            "chip_table": chip_table_data,
            "fundamental": fundamental_data, "ai_analysis": ai_data
        })
    except Exception as e:
        return jsonify({"status": "error", "message": f"內部伺服器錯誤: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))
