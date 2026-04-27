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
        # 1. 抓取股票數據
        stock = yf.Ticker(symbol)
        df = stock.history(period="1mo")
        if df.empty:
            return jsonify({"status": "error", "message": "找不到股票數據"}), 400
        
        latest_data = df.tail(10).to_string()
        # 抓取最後一個交易日的收盤價，這是模擬倉需要的關鍵數據
        current_price = float(df['Close'].iloc[-1])

        # 2. 智慧篩選模型
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        flash_15_models = [m for m in available_models if '1.5-flash' in m]
        target_model = flash_15_models[0] if flash_15_models else (available_models[0] if available_models else 'gemini-1.5-flash')

        model = genai.GenerativeModel(target_model)
        
        # 3. 升級為 15 大訊號專業提示詞
        prompt = (
            f"你是一位擁有 20 年經驗的台股操盤手。請針對 {symbol} 近期的價格與成交量數據進行「15 大訊號全面健檢」。\n"
            f"請從以下幾個維度進行結構化分析：\n"
            f"1. 均線與趨勢判定\n"
            f"2. KD、MACD 等動能指標診斷\n"
            f"3. 量價關係與潛在籌碼變化推測\n"
            f"4. 支撐壓力與近期 K 線型態\n"
            f"最後，請綜合以上訊號，給出一份包含『進場策略、停損點位』的操作建議。\n\n"
            f"數據：\n{latest_data}"
        )
        
        response = model.generate_content(prompt)
        
        return jsonify({
            "status": "success",
            "symbol": symbol,
            "current_price": current_price,
            "analysis": response.text,
            "using_model": target_model
        })
    except Exception as e:
        return jsonify({"status": "error", "message": f"系統錯誤: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
