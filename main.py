import os
from flask import Flask, request, jsonify
from flask_cors import CORS 
import yfinance as yf
import google.generativeai as genai

app = Flask(__name__)
CORS(app) 

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

        # 【關鍵修正】強制指定使用每天有 1500 次免費額度的 1.5 版本
        target_model = 'gemini-1.5-flash'
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
            "using_model": target_model
        })
    except Exception as e:
        # 如果 1.5 剛好連線不穩，自動降級到最傳統的 pro 模型 (備用方案)
        try:
            backup_model = 'gemini-pro'
            model_alt = genai.GenerativeModel(backup_model)
            response_alt = model_alt.generate_content(f"請分析 {symbol} 近期走勢，並提供 KD 與 MACD 建議:\n{latest_data}")
            return jsonify({
                "status": "success",
                "symbol": symbol,
                "analysis": response_alt.text,
                "using_model": backup_model
            })
        except Exception as backup_error:
            return jsonify({"status": "error", "message": f"主要模型錯誤: {str(e)} | 備用模型錯誤: {str(backup_error)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
