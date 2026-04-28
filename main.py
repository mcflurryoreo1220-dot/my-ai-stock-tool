import os
import json
import datetime
import traceback
import re
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

@app.route('/')
def home():
    return "AI 戰情室大腦運轉中！(搭載 3分K 與 AI 雙引擎備援)"

@app.route('/predict', methods=['GET'])
def predict():
    symbol = request.args.get('symbol', '2330.TW')
    interval = request.args.get('interval', '1d')
    
    try:
        # 新增 3m 支援
        if interval == '1d': period = "6mo"
        elif interval in ['60m', '15m']: period = "1mo"
        elif interval in ['5m', '3m']: period = "5d"
        else: period = "1mo"

        stock = yf.Ticker(symbol)
        df = stock.history(period=period, interval=interval)
        if df.empty:
            return jsonify({"status": "error", "message": f"無法獲取 {symbol} 歷史數據。"}), 400

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
            chart_data.append({
                "time": time_val, "open": round(row['Open'],2), "high": round(row['High'],2), "low": round(row['Low'],2), "close": round(row['Close'],2),
                "ma5": row['MA5'], "ma10": row['MA10'], "ma20": row['MA20'], "ma60": row['MA60']
            })
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
            if eps: fundamental_data["eps"] = round(eps, 2)
            if pe: fundamental_data["pe_ratio"] = round(pe, 2)
        except: pass

        pure_symbol = symbol.replace('.TW', '').replace('.TWO', '')
        chip_info, chip_chart_data, chip_table_data = "非日線層級", [], []
        
        if interval == '1d':
            try:
                dl = DataLoader()
                start_date = (datetime.datetime.now() - datetime.timedelta(days=45)).strftime('%Y-%m-%d')
                df_chips = dl.taiwan_stock_institutional_investors(stock_id=pure_symbol, start_date=start_date)
                if isinstance(df_chips, pd.DataFrame) and not df_chips.empty:
                    df_chips['net_buy'] = df_chips['buy'] - df_chips['sell']
                    df_chips['name'] = df_chips['name'].replace({
                        '外資及陸資(不含外資自營商)': '外資', '外資及陸資': '外資',
                        '自營商(自行買賣)': '自營', '自營商(避險)': '自營', '自營商': '自營'
                    })
                    pivot_df = df_chips.groupby(['date', 'name'])['net_buy'].sum().unstack(fill_value=0).reset_index()
                    for col in ['外資', '投信', '自營']:
                        if col not in pivot_df.columns: pivot_df[col] = 0
                    pivot_df['合計'] = pivot_df['外資'] + pivot_df['投信'] + pivot_df['自營']
                    pivot_df = pivot_df[pivot_df['合計'] != 0].copy()
                    
                    for _, r in pivot_df.iterrows():
                        chip_chart_data.append({"time": str(r['date']), "value": round(float(r['合計']) / 1000, 2)})
                    chip_info = pivot_df.tail(15).to_string() 
                    
                    last_10 = pivot_df.tail(10).iloc[::-1]
                    for _, r in last_10.iterrows():
                        chip_table_data.append({
                            "date": str(r['date'])[5:], 
                            "foreign": round(float(r['外資']) / 1000, 1),
                            "trust": round(float(r['投信']) / 1000, 1),
                            "dealer": round(float(r['自營']) / 1000, 1),
                            "total": round(float(r['合計']) / 1000, 1)
                        })
            except Exception as e:
                print("籌碼解析異常:", e)

        prompt = (
            f"你是台股頂級量化操盤手。分析 {display_name} ({interval})。\n"
            f"請輸出純 JSON，不可有 Markdown。\n"
            f"格式：\n"
            f"{{\"signal\": \"偏多/偏空/觀望\", \"pressure\": \"價格\", \"support\": \"價格\", \"stop_loss\": \"價格\", \"path_up\": \"上漲路徑\", \"path_down\": \"下跌路徑\", \"stars\": 1到5整數, \"advice\": [\"建議1\", \"建議2\", \"建議3\"]}}\n\n"
            f"基本面：{fundamental_data}\n技術面：{df.tail(20).to_string()}\n籌碼面：{chip_info}"
        )
        
        # 雙引擎備援機制
        ai_data = None
        models_to_try = ['gemini-1.5-pro-latest', 'gemini-1.5-flash']
        
        for model_name in models_to_try:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(
                    prompt, 
                    generation_config=genai.GenerationConfig(temperature=0.1, response_mime_type="application/json")
                )
                text = response.text.replace("```json", "").replace("```", "").strip()
                match = re.search(r'\{.*\}', text, re.DOTALL)
                if match:
                    ai_data = json.loads(match.group(0))
                    break # 成功解析就跳出迴圈
                else:
                    ai_data = json.loads(text)
                    break
            except Exception as e:
                print(f"[{model_name}] 引擎解析失敗，嘗試降級...")
                continue
                
        if not ai_data:
            ai_data = {"signal": "系統繁忙", "pressure": "--", "support": "--", "stop_loss": "--", "path_up": "--", "path_down": "--", "stars": 0, "advice": ["請稍後重新點擊分析"]}

        return jsonify({
            "status": "success", "symbol": symbol, "current_price": current_price, "interval": interval,
            "chart_data": chart_data, "macd_data": macd_data, "kd_data": kd_data, 
            "obv_data": obv_data, "chip_data": chip_chart_data, "chip_table": chip_table_data,
            "fundamental": fundamental_data, "ai_analysis": ai_data
        })
    except Exception as e:
        print("系統錯誤:", traceback.format_exc())
        return jsonify({"status": "error", "message": f"內部錯誤: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))
