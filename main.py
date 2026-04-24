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

        # 【智慧篩選模型邏輯】
        # 1. 向 Google 取得這把金鑰「真正能用」的所有模型清單
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        # 2. 篩選：優先找出名稱包含 '1.5-flash' 的型號 (避開 2.5 的 20 次限制)
        flash_15_models = [m for m in available_models if '1.5-flash' in m]
        
        if flash_15_models:
            # 如果有找到，就用清單裡的第一個 (例如 models/gemini-1.5-flash-latest)
            target_model = flash_15_models[0]
        else:
            # 如果真的沒有 1.5-flash，就找任何不是 2.5 的模型來墊檔
            safe_models = [m for m in available_models if '2.5' not in m]
            target_model = safe_models[0] if safe_models else available_models[0]

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
        return jsonify({"status": "error", "message": f"模型呼叫失敗，請檢查系統日誌: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
