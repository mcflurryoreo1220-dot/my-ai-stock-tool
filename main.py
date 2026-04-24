import os
from flask import Flask, request, jsonify
import yfinance as yf
import google.generativeai as genai

app = Flask(__name__)

# 設定 AI 金鑰
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

@app.route('/')
def home():
    return "AI 股票分析伺服器運行中！請在網址後方加上 /predict?symbol=2330.TW 測試。"

@app.route('/predict', methods=['GET'])
def predict():
    symbol = request.args.get('symbol', '2330.TW')
    try:
        # 1. 抓取股票數據
        stock = yf.Ticker(symbol)
        df = stock.history(period="1mo")
        if df.empty:
            return jsonify({"status": "error", "message": f"找不到股票代碼 {symbol}"}), 400
            
        latest_data = df.tail(10).to_string()
        
        # 2. 呼叫 AI 進行分析 (更換為最穩定的 gemini-pro)
        model = genai.GenerativeModel('gemini-pro')
        
        prompt = (
            f"你是一位精通技術指標的台股分析大師。請針對以下 {symbol} 的數據進行分析，"
            f"特別針對 KD 指標與 MACD 的走勢給出評價，並提供短中期的操作建議：\n\n{latest_data}"
        )
        
        response = model.generate_content(prompt)
        
        # 確保 AI 有回傳內容
        if not response.text:
            return jsonify({"status": "error", "message": "AI 未能生成回應"}), 500

        return jsonify({
            "status": "success",
            "symbol": symbol,
            "analysis": response.text
        })
    except Exception as e:
        # 詳細回報錯誤原因，方便我們除錯
        return jsonify({"status": "error", "details": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
