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

# 安全載入 API Key
api_key = os.environ.get("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

@app.route('/')
def home():
    return "AI 戰情室大腦運轉中！(搭載 JSON 強制防護與空值過濾系統)"

@app.route('/predict', methods=['GET'])
def predict():
    symbol = request.args.get('symbol', '2330.TW')
    interval = request.args.get('interval', '1d')
    
    try:
        period = "6mo" if interval == '1d' else "1mo"
        if interval == '5m': period = "5d"

        stock = yf.Ticker(symbol)
        df = stock.history(period=period, interval=interval)
        if df.empty:
            return jsonify({"status": "error", "message": f"無法獲取 {symbol} 的歷史價量數據。"}), 400

        # 計算均線
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA10'] = df['Close'].rolling(window=10).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA60'] = df['Close'].rolling(window=60).mean()

        # MACD & KD
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
        
        # OBV
        df['Volume_Dir'] = np.sign(df['Close'].diff()).fillna(0)
        df['OBV'] = (df['Volume'] * df['Volume_Dir']).cumsum()

        df = df.fillna(0)
        df_chart = df.tail(80)
        
        chart_data, macd_data, kd_data, obv_data = [], [], [], []
        for date, row in df_chart.iterrows():
            time_val = date.strftime('%Y-%m-%d') if interval == '1d' else int(date.timestamp())
            chart_data.append({
                "time": time_val, "open": row['Open'], "high": row['High'], "low": row['Low'], "close": row['Close'],
                "ma5": row['MA5'], "ma10": row['MA10'], "ma20": row['MA20'], "ma60": row['MA60']
            })
            macd_data.append({"time": time_val, "dif": row['DIF'], "signal": row['MACD_Signal'], "osc": row['OSC']})
            kd_data.append({"time": time_val, "k": row['K'], "d": row['D']})
            obv_data.append({"time": time_val, "value": row['OBV']})

        current_price = float(df['Close'].iloc[-1])
        display_name = symbol

        # === 【保險絲 C：基本面雙重抓取機制】 ===
        fundamental_data = {"eps": "--", "pe_ratio": "--", "dividend_yield": "--", "market_cap": "--"}
        try:
            info = stock.info
            display_name = info.get('shortName', symbol)
            
            eps = info.get("trailingEps")
            pe = info.get("trailingPE")
            dy = info.get("dividendYield")
            mc = info.get("marketCap")
            
            # 若 info 抓不到，嘗試用 fast_info 補救市值
            if mc is None and hasattr(stock, 'fast_info'):
                mc = stock.fast_info.get('market_cap')

            fundamental_data["eps"] = eps if eps is not None else "--"
            fundamental_data["pe_ratio"] = pe if pe is not None else "--"
            fundamental_data["dividend_yield"] = f"{round(dy * 100, 2)}%" if dy else "--"
            fundamental_data["market_cap"] = f"{round(mc / 1e12, 2)} 兆" if mc else "--"
        except Exception as info_err:
            print(f"[{symbol}] Yahoo 基本面解析異常: {info_err}")

        # === 【保險絲 B：精細化籌碼，剔除 0 值空包彈】 ===
        pure_symbol = symbol.replace('.TW', '').replace('.TWO', '')
        chip_info, chip_chart_data, chip_table_data = "非日線層級或無資料", [], []
        
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
                    
                    # 【核心過濾】：剔除三大法人合計為 0 的假日/無交易日
                    pivot_df = pivot_df[pivot_df['外資'] != 0].copy()
                    
                    # 圖表數據
                    for _, r in pivot_df.iterrows():
                        chip_chart_data.append({"time": str(r['date']), "value": round(float(r['合計']) / 1000, 2)})
                    
                    chip_info = pivot_df.tail(15).to_string() 
                    
                    # 表格數據 (取近 10 筆有效交易日)
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

        # === 【保險絲 A：PRO 旗艦大腦強制 JSON 輸出】 ===
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        pro_models = [m for m in available_models if '1.5-pro' in m]
        target_model = pro_models[0] if pro_models else 'gemini-1.5-pro-latest'
        model = genai.GenerativeModel(target_model)
        
        prompt = (
            f"你是台股頂級量化操盤手。請分析 {display_name} ({interval})。\n"
            f"必須只輸出純 JSON，不可有 Markdown。\n"
            f"JSON 格式嚴格規定：\n"
            f"1. \"signal\": 4字以內\n2. \"pressure\": 價格\n3. \"support\": 價格\n"
            f"4. \"stop_loss\": 價格\n5. \"path_up\": 預測路徑\n6. \"path_down\": 預測路徑\n"
            f"7. \"stars\": 1-5整數\n8. \"advice\": 3個建議陣列\n\n"
            f"【基本面數據】：{fundamental_data}\n"
            f"【技術面數據】：{df.tail(20).to_string()}\n"
            f"【籌碼面明細】：{chip_info}"
        )
        
        try:
            # 強制宣告 response_mime_type 為 application/json，確保 AI 絕對不會亂加 Markdown
            response = model.generate_content(
                prompt, 
                generation_config=genai.GenerationConfig(
                    temperature=0.1,
                    response_mime_type="application/json"
                )
            )
            text = response.text
            
            # 雙重防護 Regex 提取
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                ai_data = json.loads(match.group(0))
            else:
                ai_data = json.loads(text)
                
        except Exception as ai_err:
            print("AI 解析錯誤:", traceback.format_exc())
            ai_data = {
                "signal": "解析異常", "pressure": "--", "support": "--", "stop_loss": "--", 
                "path_up": "資料格式錯誤", "path_down": "資料格式錯誤", "stars": 0, 
                "advice": ["AI 格式解析失敗", "圖表與籌碼已載入", "請稍後重試"]
            }

        return jsonify({
            "status": "success", "symbol": symbol, "current_price": current_price, "interval": interval,
            "chart_data": chart_data, "macd_data": macd_data, "kd_data": kd_data, 
            "obv_data": obv_data, "chip_data": chip_chart_data, "chip_table": chip_table_data,
            "fundamental": fundamental_data, "ai_analysis": ai_data
        })
    except Exception as e:
        print("系統嚴重錯誤:", traceback.format_exc())
        return jsonify({"status": "error", "message": f"內部伺服器錯誤: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))
