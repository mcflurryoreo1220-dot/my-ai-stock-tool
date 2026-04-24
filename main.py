import os
from flask import Flask, request, jsonify
from flask_cors import CORS  # 為了讓網頁能抓到資料
import yfinance as yf
import google.generativeai as genai

app = Flask(__name__)
CORS(app) # 允許前端網頁存取

genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

@app.route('/')
def home():
    return "伺服器已啟動！"

@app.route('/predict', methods=['GET'])
def predict():
    symbol = request.args.get('symbol', '2330.TW')
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period="1mo")
        latest_data = df.tail(10).to_string()

        # 自動尋找目前可用的模型名稱
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        model_name = available_models[0] if available_models else 'models/gemini-1.5-flash'

        model = genai.GenerativeModel(model_name)
        prompt = f"你是一位台股技術分析專家，請分析以下數據的 KD 與 MACD 趨勢：\n{latest_data}"
        response = model.generate_content(prompt)
        
        return jsonify({
            "status": "success",
            "symbol": symbol,
            "analysis": response.text
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))
