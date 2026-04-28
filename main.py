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

# 確保安全載入 API Key
api_key = os.environ.get("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

@app.route('/')
def home():
    return "AI 戰情室大腦運轉中！(搭載 OBV 與多週期運算引擎)"

@app.route('/predict', methods=['GET'])
def predict():
    symbol = request.args.get('symbol', '2330.TW')
    # 接收前端傳來的时间週期 (預設為 1d 日線)
    interval = request.args.get('interval', '1d')
    
    try:
        # 根據週期動態調整抓取長度
        if interval == '1d':
            period = "6mo"
        elif interval == '60m':
            period = "1mo"
        elif interval == '15m':
            period = "1mo"
        elif interval == '5m':
            period = "5d"
        else:
            period = "1mo"

        # 1. 抓取技術面數據
        stock = yf.Ticker(symbol)
        df = stock.history(period=period, interval=interval)
        if df.empty:
            return jsonify({"status": "error", "message": f"無法獲取 {symbol} 數據。台灣時間盤中或剛開盤時，分K資料可能會有延遲。"}), 400

        # 計算 MACD 與 KD
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
        df['K'] = K
        df['D'] = D
        
        # 【新增方向一】：計算 OBV (能量潮指標)
        df['Volume_Dir'] = np.sign(df['Close'].diff())
        df['Volume_Dir'] = df['Volume_Dir'].fillna(0)
        df['OBV'] = (df['Volume'] * df['Volume_Dir']).cumsum()

        df = df.fillna(0)
        # 取最後 80 筆資料畫圖，讓分K能看多一點
        df_chart = df.tail(80)
        
        chart_data, macd_data, kd_data, obv_data = [], [], [], []
        for date, row in df_chart.iterrows():
            # 處理時間格式：日線用字串，分K用 Unix Timestamp 確保圖表時間軸精準
            if interval == '1d':
                time_val = date.strftime('%Y-%m-%d')
            else:
                time_val = int(date.timestamp())

            chart_data.append({"time": time_val, "open": row['Open'], "high": row['High'], "low": row['Low'], "close": row['Close']})
            macd_data.append({"time": time_val, "dif": row['DIF'], "signal": row['MACD_Signal'], "osc": row['OSC']})
            kd_data.append({"time": time_val, "k": row['K'], "d": row['D']})
            obv_data.append({"time": time_val, "value": row['OBV']})

        latest_data = df.tail(40).to_string()
        current_price = float(df['Close'].iloc[-1])

        try:
            company_name = stock.info.get('shortName', '')
            display_name = f"{company_name} ({symbol})" if company_name else symbol
        except:
            display_name = symbol
            
        # 抓取籌碼數據 (僅日線才有三大法人)
        pure_symbol = symbol.replace('.TW', '').replace('.TWO', '')
        chip_info = "非日線層級，略過籌碼分析"
        chip_chart_data = []
        
        if interval == '1d':
            try:
                dl = DataLoader()
                start_date = (datetime.datetime.now() - datetime.timedelta(days=90)).strftime('%Y-%m-%d')
                df_chips = dl.taiwan_stock_institutional_investors(stock_id=pure_symbol, start_date=start_date)
                if isinstance(df_chips, pd.DataFrame) and not df_chips.empty:
                    df_chips['net_buy'] = df_chips['buy'] - df_chips['sell']
                    chip_info = df_chips[['date', 'name', 'net_buy']].tail(40).to_string()
                    daily_chips = df_chips.groupby('date')['net_buy'].sum().reset_index()
                    for _, r in daily_chips.iterrows():
                        chip_chart_data.append({"time": str(r['date']), "value": round(float(r['net_buy']) / 1000, 2)})
            except Exception as e:
                pass

        # ==========================================
        # 2. 升級為 PRO 旗艦大腦核心
        # ==========================================
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        pro_models = [m for m in available_models if '1.5-pro' in m]
        target_model = pro_models[0] if pro_models else 'gemini-1.5-pro-latest'
        model = genai.GenerativeModel(target_model)
        
        interval_name = "日線" if interval == "1d" else f"{interval} 分線"

        # 3. 藍圖級 AI 提示詞 (加入多週期認知)
        prompt = (
            f"你是台股頂級量化操盤手。請針對 {display_name} 的『{interval_name}』級別數據進行分析。\n"
            f"必須只輸出純 JSON，不可有 Markdown。\n"
            f"JSON 格式嚴格規定：\n"
            f"1. \"signal\": 4字以內，如「短線偏多」或「觀望」\n"
            f"2. \"pressure\": 數字或短字串，如「220.5」\n"
            f"3. \"support\": 數字或短字串，如「195.0」\n"
            f"4. \"stop_loss\": 數字或短字串，如「190.0」\n"
            f"5. \"path_up\": 若上漲的可能路徑(限15字)\n"
            f"6. \"path_down\": 若下跌的可能路徑(限15字)\n"
            f"7. \"stars\": 1到5的整數，代表綜合推薦星級\n"
            f"8. \"advice\": 陣列包含3個字串，每個字串為一句簡短操作建議(限15字)\n\n"
            f"【技術面 ({interval_name} 與 OBV量能)】：\n{latest_data}\n\n"
            f"【籌碼面】：\n{chip_info}"
        )
        
        try:
            response = model.generate_content(prompt, generation_config=genai.GenerationConfig(temperature=0.1))
            text = response.text
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1:
                ai_data = json.loads(text[start:end+1])
            else:
                raise ValueError("JSON格式錯誤")
        except Exception as e:
            print("AI 解析錯誤:", e)
            ai_data = {
                "signal": "解析失敗", "pressure": "--", "support": "--", "stop_loss": "--",
                "path_up": "資料異常", "path_down": "資料異常", "stars": 0,
                "advice": ["伺服器繁忙", "請稍後重試", "圖表已載入"]
            }
        
        return jsonify({
            "status": "success", "symbol": symbol, "current_price": current_price, "interval": interval,
            "chart_data": chart_data, "macd_data": macd_data, "kd_data": kd_data, 
            "obv_data": obv_data, "chip_data": chip_chart_data,
            "ai_analysis": ai_data,
            "using_model": target_model
        })
    except Exception as e:
        print("系統錯誤:", traceback.format_exc())
        return jsonify({"status": "error", "message": f"內部錯誤: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
