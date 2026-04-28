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

api_key = os.environ.get("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

@app.route('/')
def home():
    return "AI 戰情室大腦運轉中！(搭載均線與籌碼明細引擎)"

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
            return jsonify({"status": "error", "message": f"無法獲取 {symbol} 數據。"}), 400

        # === 【新增：計算均線 MA5, MA10, MA20, MA60】 ===
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
            # 將 MA 數據包入 chart_data
            chart_data.append({
                "time": time_val, "open": row['Open'], "high": row['High'], "low": row['Low'], "close": row['Close'],
                "ma5": row['MA5'], "ma10": row['MA10'], "ma20": row['MA20'], "ma60": row['MA60']
            })
            macd_data.append({"time": time_val, "dif": row['DIF'], "signal": row['MACD_Signal'], "osc": row['OSC']})
            kd_data.append({"time": time_val, "k": row['K'], "d": row['D']})
            obv_data.append({"time": time_val, "value": row['OBV']})

        current_price = float(df['Close'].iloc[-1])
        display_name = stock.info.get('shortName', symbol)

        info = stock.info
        fundamental_data = {
            "eps": info.get("trailingEps", "--"),
            "pe_ratio": info.get("trailingPE", "--"),
            "dividend_yield": f"{round(info.get('dividendYield', 0) * 100, 2)}%" if info.get('dividendYield') else "--",
            "market_cap": f"{round(info.get('marketCap', 0) / 1e12, 2)} 兆" if info.get('marketCap') else "--"
        }

        # === 【新增：精細化籌碼明細運算】 ===
        pure_symbol = symbol.replace('.TW', '').replace('.TWO', '')
        chip_info, chip_chart_data, chip_table_data = "非日線層級", [], []
        
        if interval == '1d':
            try:
                dl = DataLoader()
                start_date = (datetime.datetime.now() - datetime.timedelta(days=40)).strftime('%Y-%m-%d')
                df_chips = dl.taiwan_stock_institutional_investors(stock_id=pure_symbol, start_date=start_date)
                if isinstance(df_chips, pd.DataFrame) and not df_chips.empty:
                    df_chips['net_buy'] = df_chips['buy'] - df_chips['sell']
                    # 簡化法人名稱以利彙整
                    df_chips['name'] = df_chips['name'].replace({
                        '外資及陸資(不含外資自營商)': '外資', '外資及陸資': '外資',
                        '自營商(自行買賣)': '自營', '自營商(避險)': '自營', '自營商': '自營'
                    })
                    
                    # 製作圖表用的總和數據
                    daily_total = df_chips.groupby('date')['net_buy'].sum().reset_index()
                    for _, r in daily_total.iterrows():
                        chip_chart_data.append({"time": str(r['date']), "value": round(float(r['net_buy']) / 1000, 2)})
                    
                    # 製作表格用的分列數據 (樞紐分析)
                    pivot_df = df_chips.groupby(['date', 'name'])['net_buy'].sum().unstack(fill_value=0).reset_index()
                    for col in ['外資', '投信', '自營']:
                        if col not in pivot_df.columns: pivot_df[col] = 0
                    pivot_df['合計'] = pivot_df['外資'] + pivot_df['投信'] + pivot_df['自營']
                    
                    chip_info = pivot_df.tail(20).to_string() # 給 AI 看的
                    
                    # 給前端畫表格用的 (取近 10 日，反轉順序讓最新日在最上面)
                    last_10 = pivot_df.tail(10).iloc[::-1]
                    for _, r in last_10.iterrows():
                        chip_table_data.append({
                            "date": str(r['date'])[5:], # 只取 MM-DD
                            "foreign": round(float(r['外資']) / 1000, 1),
                            "trust": round(float(r['投信']) / 1000, 1),
                            "dealer": round(float(r['自營']) / 1000, 1),
                            "total": round(float(r['合計']) / 1000, 1)
                        })
            except Exception as e:
                print("籌碼解析異常:", e)

        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        pro_models = [m for m in available_models if '1.5-pro' in m]
        target_model = pro_models[0] if pro_models else 'gemini-1.5-pro-latest'
        model = genai.GenerativeModel(target_model)
        
        prompt = (
            f"你是台股頂級量化操盤手。請分析 {display_name} ({interval})。\n"
            f"JSON 格式嚴格規定：\n"
            f"1. \"signal\": 4字以內\n2. \"pressure\": 價格\n3. \"support\": 價格\n"
            f"4. \"stop_loss\": 價格\n5. \"path_up\": 預測路徑\n6. \"path_down\": 預測路徑\n"
            f"7. \"stars\": 1-5整數\n8. \"advice\": 3個建議陣列\n\n"
            f"【基本面數據】：{fundamental_data}\n"
            f"【技術面數據】：{df.tail(20).to_string()}\n"
            f"【籌碼面明細】：{chip_info}"
        )
        
        try:
            response = model.generate_content(prompt, generation_config=genai.GenerationConfig(temperature=0.1))
            text = response.text
            start, end = text.find('{'), text.rfind('}')
            ai_data = json.loads(text[start:end+1])
        except:
            ai_data = {"signal": "解析失敗", "stars": 0, "advice": ["AI 繁忙", "請重試"]}

        return jsonify({
            "status": "success", "symbol": symbol, "current_price": current_price, "interval": interval,
            "chart_data": chart_data, "macd_data": macd_data, "kd_data": kd_data, 
            "obv_data": obv_data, "chip_data": chip_chart_data, "chip_table": chip_table_data,
            "fundamental": fundamental_data, "ai_analysis": ai_data
        })
    except Exception as e:
        print("系統錯誤:", traceback.format_exc())
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))
