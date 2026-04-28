import os
import json
import datetime
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS 
import yfinance as yf
import pandas as pd
import numpy as np  # 新增：用於處理無限大等數學防呆
import google.generativeai as genai
from FinMind.data import DataLoader

app = Flask(__name__)
CORS(app) 

genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

@app.route('/')
def home():
    return "AI 戰情室大腦運轉中！(具備工業級防護)"

@app.route('/predict', methods=['GET'])
def predict():
    symbol = request.args.get('symbol', '2330.TW')
    try:
        # 1. 抓取技術面數據
        stock = yf.Ticker(symbol)
        df = stock.history(period="6mo")
        if df.empty:
            return jsonify({"status": "error", "message": f"無法從資料庫獲取 {symbol} 的數據，請確認代碼或稍後再試。"}), 400

        # 計算 MACD 與 KD
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['DIF'] = df['EMA12'] - df['EMA26']
        df['MACD_Signal'] = df['DIF'].ewm(span=9, adjust=False).mean()
        df['OSC'] = df['DIF'] - df['MACD_Signal']

        df['9_high'] = df['High'].rolling(9).max()
        df['9_low'] = df['Low'].rolling(9).min()
        
        # 【防呆機制】：避免股價不動時，除以零產生無限大 (inf) 導致系統崩潰
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
        
        df['K'] = K
        df['D'] = D
        
        # 強制填補所有空值，確保 JSON 格式正確
        df = df.fillna(0)
        df_chart = df.tail(60)
        
        chart_data, macd_data, kd_data = [], [], []
        for date, row in df_chart.iterrows():
            time_str = date.strftime('%Y-%m-%d')
            chart_data.append({"time": time_str, "open": row['Open'], "high": row['High'], "low": row['Low'], "close": row['Close']})
            macd_data.append({"time": time_str, "dif": row['DIF'], "signal": row['MACD_Signal'], "osc": row['OSC']})
            kd_data.append({"time": time_str, "k": row['K'], "d": row['D']})

        latest_data = df.tail(40).to_string()
        current_price = float(df['Close'].iloc[-1])

        try:
            company_name = stock.info.get('shortName', '')
            display_name = f"{company_name} ({symbol})" if company_name else symbol
        except:
            display_name = symbol
            
        # 籌碼數據抓取
        pure_symbol = symbol.replace('.TW', '').replace('.TWO', '')
        chip_info = "無法取得籌碼資料"
        chip_chart_data = []
        try:
            dl = DataLoader()
            start_date = (datetime.datetime.now() - datetime.timedelta(days=90)).strftime('%Y-%m-%d')
            df_chips = dl.taiwan_stock_institutional_investors(stock_id=pure_symbol, start_date=start_date)
            
            if isinstance(df_chips, pd.DataFrame) and not df_chips.empty:
                df_chips['net_buy'] = df_chips['buy'] - df_chips['sell']
                chip_info = df_chips[['date', 'name', 'net_buy']].tail(40).to_string()
                daily_chips = df_chips.groupby('date')['net_buy'].sum().reset_index()
                for _, r in daily_chips.iterrows():
                    chip_chart_data.append({
                        "time": str(r['date']),
                        "value": round(float(r['net_buy']) / 1000, 2)
                    })
        except Exception as e:
            print("籌碼抓取失敗:", e)

        # 2. 智慧篩選模型
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        flash_15_models = [m for m in available_models if '1.5-flash' in m]
        target_model = flash_15_models[0] if flash_15_models else (available_models[0] if available_models else 'gemini-1.5-flash')
        model = genai.GenerativeModel(target_model)
        
        prompt = (
            f"你是一位擁有 20 年經驗的台股操盤手。請針對 {display_name} 的數據進行分析。\n"
            f"必須以純 JSON 格式輸出結果，絕不包含 Markdown 標記。\n"
            f"Key 包含：trend, pressure, support, summary, action。\n\n"
            f"技術面：\n{latest_data}\n\n"
            f"籌碼面：\n{chip_info}"
        )
        
        # 【防呆機制】：AI 當機時的安全網
        try:
            response = model.generate_content(prompt)
            raw_text = response.text.replace("```json", "").replace("```", "").strip()
            ai_data = json.loads(raw_text)
        except Exception as ai_err:
            print("AI 解析錯誤:", ai_err)
            ai_data = {
                "trend": "AI 暫停服務", 
                "pressure": 0, 
                "support": 0, 
                "summary": "Google AI 伺服器目前繁忙或回傳格式異常，但圖表數據已為您載入。", 
                "action": "請稍後重新點擊分析"
            }
        
        return jsonify({
            "status": "success",
            "symbol": symbol,
            "current_price": current_price,
            "chart_data": chart_data,
            "macd_data": macd_data,
            "kd_data": kd_data,
            "chip_data": chip_chart_data,
            "ai_analysis": ai_data
        })
    except Exception as e:
        # 將真實錯誤印在 Render 後台，並把具體原因傳給前端網頁
        error_details = traceback.format_exc()
        print("伺服器嚴重錯誤:\n", error_details)
        return jsonify({"status": "error", "message": f"內部錯誤: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
