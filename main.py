import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS 
import yfinance as yf
import pandas as pd
import google.generativeai as genai

app = Flask(__name__)
CORS(app) 

genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

@app.route('/')
def home():
    return "AI 戰情室大腦運轉中！"

@app.route('/predict', methods=['GET'])
def predict():
    symbol = request.args.get('symbol', '2330.TW')
    try:
        # 1. 抓取股票數據 (拉長到 6 個月確保均線與 MACD 計算準確)
        stock = yf.Ticker(symbol)
        df = stock.history(period="6mo")
        if df.empty:
            return jsonify({"status": "error", "message": "找不到股票數據"}), 400

        # === 【新增技術指標運算引擎】 ===
        # 計算 MACD (12, 26, 9)
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['DIF'] = df['EMA12'] - df['EMA26']
        df['MACD_Signal'] = df['DIF'].ewm(span=9, adjust=False).mean()
        df['OSC'] = df['DIF'] - df['MACD_Signal'] # 柱狀圖

        # 計算 KD (9, 3, 3) - 台股標準算法
        df['9_high'] = df['High'].rolling(9).max()
        df['9_low'] = df['Low'].rolling(9).min()
        df['RSV'] = (df['Close'] - df['9_low']) / (df['9_high'] - df['9_low']) * 100
        
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
        # ==============================

        # 將含有空值的早期資料濾除，保留近 3 個月畫圖
        df = df.dropna().tail(60) 

        # 打包 K 線與指標資料
        chart_data = []
        macd_data = []
        kd_data = []
        
        for date, row in df.iterrows():
            time_str = date.strftime('%Y-%m-%d')
            chart_data.append({"time": time_str, "open": row['Open'], "high": row['High'], "low": row['Low'], "close": row['Close']})
            # MACD 包含 DIF(快線), Signal(慢線), OSC(柱狀圖)
            macd_data.append({"time": time_str, "dif": row['DIF'], "signal": row['MACD_Signal'], "osc": row['OSC']})
            # KD 包含 K與D
            kd_data.append({"time": time_str, "k": row['K'], "d": row['D']})

        latest_data = df.tail(40).to_string()
        current_price = float(df['Close'].iloc[-1])

        try:
            company_name = stock.info.get('shortName', '')
            display_name = f"{company_name} ({symbol})" if company_name else symbol
        except:
            display_name = symbol

        # 2. 智慧篩選模型
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        flash_15_models = [m for m in available_models if '1.5-flash' in m]
        target_model = flash_15_models[0] if flash_15_models else (available_models[0] if available_models else 'gemini-1.5-flash')
        model = genai.GenerativeModel(target_model)
        
        # 3. AI 結構化提示詞
        prompt = (
            f"你是一位擁有 20 年經驗的台股操盤手。請針對 {display_name} 的數據進行分析。\n"
            f"【極重要指示】：\n"
            f"你必須以純 JSON 格式輸出結果，絕對不要包含任何 Markdown 標記。\n"
            f"JSON 的 Key 必須嚴格包含以下五個項目：\n"
            f"1. \"trend\" (字串，判斷目前趨勢)\n"
            f"2. \"pressure\" (數字，近期的關鍵壓力價位)\n"
            f"3. \"support\" (數字，近期的關鍵支撐價位)\n"
            f"4. \"summary\" (字串，100字以內的技術面與K線型態總結)\n"
            f"5. \"action\" (字串，具體的操作建議)\n\n"
            f"數據：\n{latest_data}"
        )
        
        response = model.generate_content(prompt)
        
        try:
            raw_text = response.text.replace("```json", "").replace("```", "").strip()
            ai_data = json.loads(raw_text)
        except:
            ai_data = {"trend": "解析失敗", "pressure": 0, "support": 0, "summary": "AI 回傳異常。", "action": "暫停操作"}
        
        return jsonify({
            "status": "success",
            "symbol": symbol,
            "current_price": current_price,
            "chart_data": chart_data,
            "macd_data": macd_data,
            "kd_data": kd_data,
            "ai_analysis": ai_data
        })
    except Exception as e:
        return
