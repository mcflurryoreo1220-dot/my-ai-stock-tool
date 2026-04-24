import os
from flask import Flask, request, jsonify
import yfinance as yf
import google.generativeai as genai

app = Flask(__name__)

# 設定 AI 金鑰
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

# 【新增】大門口接待處：讓 Render 知道這台機器運作正常
@app.route('/')
def home():
    return "AI 股票分析伺服器運行中！請使用 /predict 端點進行查詢。"

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
        model = genai.GenerativeModel('gemini-1.5-flash') # 使用更新、更快的模型
        prompt = (
            f"你是一位精通台股技術指標的專家。請針對 {symbol} 的數據進行分析。"
            f"請重點分析 KD 指標與 MACD 趨勢，並給出專業建議：\n\n{latest_data}"
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
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
