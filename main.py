import os
import json
import datetime
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS 
import yfinance as yf
import pandas as pd
import numpy as np
import google.generativeai as genai
from FinMind.data import DataLoader

app = Flask(__name__)
CORS(app) 

# 安全載入 API Key
api_key = os.environ.get("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

@app.route('/')
def home():
    return "AI 戰情室大腦運轉中！(基本面模組 A 計畫已上線)"

@app.route('/predict', methods=['GET'])
def predict():
    symbol = request.args.get('symbol', '2330.TW')
    interval = request.args.get('interval', '1d')
    
    try:
        # 動態調整抓取長度
        period = "6mo" if interval == '1d' else "1mo"
        if interval == '5m': period = "5d"

        # 1. 抓取技術面數據
        stock = yf.Ticker(symbol)
        df = stock.history(period=period, interval=interval)
        if df.empty:
            return jsonify({"status": "error", "message": f"無法獲取 {symbol} 數據。"}), 400

        # === 技術指標運算 ===
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
        
        # OBV 能量潮
        df['Volume_Dir'] = np.sign(df['Close'].diff()).fillna(0)
        df['OBV'] = (df['Volume'] * df['Volume_Dir']).cumsum()

        df_chart = df.tail(80)
        chart_data, macd_data, kd_data, obv_data = [], [], [], []
        for date, row in df_chart.iterrows():
            time_val = date.strftime('%Y-%m-%d') if interval == '1d' else int(date.timestamp())
            chart_data.append({"time": time_val, "open": row['Open'], "high": row['High'], "low": row['Low'], "close": row['Close']})
            macd_data.append({"time": time_val, "dif": row['DIF'], "signal": row['MACD_Signal'], "osc": row['OSC']})
            kd_data.append({"time": time_val, "k": row['K'], "d": row['D']})
            obv_data.append({"time": time_val, "value": row['OBV']})

        current_price = float(df['Close'].iloc[-1])
        display_name = stock.info.get('shortName', symbol)

        # === 【A 計畫：基本面數據採購】 ===
        info = stock.info
        fundamental_data = {
            "eps": info.get("trailingEps", "--"),
            "pe_ratio": info.get("trailingPE", "--"),
            "dividend_yield": f"{round(info.get('dividendYield', 0) * 100, 2)}%" if info.get('dividendYield') else "--",
            "market_cap": f"{round(info.get('marketCap', 0) / 1e12, 2)} 兆" if info.get('marketCap') else "--"
        }

        # 抓取日線籌碼
        pure_symbol = symbol.replace('.TW', '').replace('.TWO', '')
        chip_info, chip_chart_data = "非日線層級", []
        if interval == '1d':
            try:
                dl = DataLoader()
                start_date = (datetime.datetime.now() - datetime.timedelta(days=90)).strftime('%Y-%m-%d')
                df_chips = dl.taiwan_stock_institutional_investors(stock_id=pure_symbol, start_date=start_date)
                if not df_chips.empty:
                    df_chips['net_buy'] = df_chips['buy'] - df_chips['sell']
                    chip_info = df_chips[['date', 'name', 'net_buy']].tail(30).to_string()
                    daily_chips = df_chips.groupby('date')['net_buy'].sum().reset_index()
                    for _, r in daily_chips.iterrows():
                        chip_chart_data.append({"time": str(r['date']), "value": round(float(r['net_buy']) / 1000, 2)})
            except: pass

        # 2. PRO 旗艦大腦分析
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        prompt = (
            f"你是台股頂級量化操盤手。請分析 {display_name} ({interval})。\n"
            f"JSON 格式嚴格規定：\n"
            f"1. \"signal\": 4字以內\n2. \"pressure\": 價格\n3. \"support\": 價格\n"
            f"4. \"stop_loss\": 價格\n5. \"path_up\": 預測路徑\n6. \"path_down\": 預測路徑\n"
            f"7. \"stars\": 1-5整數\n8. \"advice\": 3個建議陣列\n\n"
            f"【基本面數據】：{fundamental_data}\n"
            f"【技術面數據】：{df.tail(30).to_string()}\n"
            f"【籌碼面數據】：{chip_info}"
        )
        
        try:
            response = model.generate_content(prompt, generation_config=genai.GenerationConfig(temperature=0.1))
            text = response.text
            start, end = text.find('{'), text.rfind('}')
            ai_data = json.loads(text[start:end+1])
        except:
            ai_data = {"signal": "解析失敗", "stars": 0, "advice": ["AI 繁忙", "請重試"]}

        return jsonify({
            "status": "success", "symbol": symbol, "current_price": current_price,
            "chart_data": chart_data, "macd_data": macd_data, "kd_data": kd_data, 
            "obv_data": obv_data, "chip_data": chip_chart_data,
            "fundamental": fundamental_data, "ai_analysis": ai_data
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))
