import os
from flask import Flask, request, jsonify
import yfinance as yf
import google.generativeai as genai

app = Flask(__name__)

# 設定您的 AI 金鑰 (之後會在 Render 設定環境變數)
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

@app.route('/predict', methods=['GET'])
def predict():
    # 預設查詢台積電，也可以透過網址參數更換
    symbol = request.args.get('symbol', '2330.TW')
    
    try:
        # 1. 抓取股票數據 (最近一個月)
        stock = yf.Ticker(symbol)
        df = stock.history(period="1mo")
        
        if df.empty:
            return jsonify({"error": "找不到股票數據"}), 400
            
        latest_data = df.tail(10).to_string() # 取最後 10 筆資料
        
        # 2. 呼叫 Gemini AI 進行分析
        model = genai.GenerativeModel('gemini-pro')
        prompt = (
            f"你是一位精通技術指標的台股分析師。請針對以下 {symbol} 的近期數據，"
            f"分析 KD、MACD 的趨勢，並給出專業的投資建議與風險提示：\n\n{latest_data}"
        )
        response = model.generate_content(prompt)
        
        return jsonify({
            "symbol": symbol,
            "analysis": response.text
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Render 要求的連接埠設定
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
