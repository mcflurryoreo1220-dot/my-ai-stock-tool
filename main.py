import os
from flask import Flask, request, jsonify
from flask_cors import CORS 
import yfinance as yf
import google.generativeai as genai

app = Flask(__name__)
CORS(app) # 這行是為了讓網頁能順利抓資料

# 配置 API Key
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

@app.route('/')
def home():
    return "AI 股票分析基地台已連線！"

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

        # 2. 自動尋找目前可用的模型 (解決 404 問題)
        # 我們先試著抓出所有支援產生成內容的模型
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        # 優先使用 flash 模型，若沒有則用清單第一個
        target_model = 'models/gemini-1.5-flash'
        if target_model not in available_models:
            target_model = available_models[0] if available_models else 'gemini-pro'

        model = genai.GenerativeModel(target_model)
        
        prompt = (
            f"你是一位精通技術指標的台股分析師。請針對 {symbol} 數據分析 KD 指標與 MACD 趨勢，"
            f"並給出操作建議：\n\n{latest_data}"
        )
        
        response = model.generate_content(prompt)
        
        return jsonify({
            "status": "success",
            "symbol": symbol,
            "analysis": response.text,
            "using_model": target_model # 讓您知道它用了哪個模型
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
