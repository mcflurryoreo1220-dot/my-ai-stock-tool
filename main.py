import os
from flask import Flask, request, jsonify
import yfinance as yf
import google.generativeai as genai

app = Flask(__name__)

# 設定 AI 金鑰
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

# 【關鍵修正】大門接待處：讓 Render 知道程式運作正常
@app.route('/')
def home():
    return "AI 股票分析伺服器運行中！請在網址後方加上 /predict?symbol=2330.TW 進行測試。"

@app.route('/predict', methods=['GET'])
def predict():
    symbol = request.args.get('symbol', '2330.TW')
    try:
        # 1. 抓取股票數據
        stock = yf.Ticker(symbol)
        df = stock.history(period="1mo")
        if df.empty:
            return jsonify({"error": "找不到股票數據"}), 400
            
        latest_data = df.tail(10).to_string()
        
        # 2. 呼叫 AI 進行分析
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = (
            f"你是一位精通台股技術指標的專家。請針對 {symbol} 的近期數據進行分析。"
            f"請重點分析 KD 指標與 MACD 趨勢，並給出專業的投資建議：\n\n{latest_data}"
        )
        response = model.generate_content(prompt)
        
        return jsonify({
            "status": "success",
            "symbol": symbol,
            "analysis": response.text
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    # Render 預設使用 10000 端口
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
