import os
from flask import Flask, request, jsonify
from flask_cors import CORS 
import yfinance as yf
import google.generativeai as genai

app = Flask(__name__)
CORS(app) 

genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

@app.route('/')
def home():
    return "工業級 AI 股票分析基地台運轉中！"

@app.route('/predict', methods=['GET'])
def predict():
    symbol = request.args.get('symbol', '2330.TW')
    try:
        # 1. 抓取股票數據 (3個月數據)
        stock = yf.Ticker(symbol)
        df = stock.history(period="3mo")
        if df.empty:
            return jsonify({"status": "error", "message": "找不到股票數據"}), 400
        
        # 【新增功能】：打包 K 線圖表所需的資料
        chart_data = []
        for date, row in df.iterrows():
            chart_data.append({
                "time": date.strftime('%Y-%m-%d'),
                "open": float(row['Open']),
                "high": float(row['High']),
                "low": float(row['Low']),
                "close": float(row['Close'])
            })

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
        
        # 3. AI 提示詞
        prompt = (
            f"你是一位擁有 20 年經驗的台股操盤手。請針對 {display_name} 的數據進行「15 大訊號全面健檢」。\n"
            f"【重要指示】：\n"
            f"1. 若 {display_name} 包含英文名稱，請自動替換為台灣股民熟知的「中文簡稱」。\n"
            f"2. 我已提供近 40 個交易日的完整數據，請直接進行均線、KD、MACD 等實質分析，絕對不要在報告中出現「數據不足」等推託之詞。\n"
            f"3. 請保持專業、俐落的市場老手語氣。\n\n"
            f"請從以下維度分析：一、均線與趨勢判定；二、動能指標診斷；三、量價關係；四、支撐壓力與K線型態。\n"
            f"最後給出包含『進場策略、停損點位』的操作建議。\n\n"
            f"數據：\n{latest_data}"
        )
        
        response = model.generate_content(prompt)
        
        return jsonify({
            "status": "success",
            "symbol": symbol,
            "current_price": current_price,
            "chart_data": chart_data, # 將圖表資料傳給網頁
            "analysis": response.text,
            "using_model": target_model
        })
    except Exception as e:
        return jsonify({"status": "error", "message": f"系統錯誤: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
