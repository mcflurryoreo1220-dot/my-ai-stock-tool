import os
from flask import Flask, request, jsonify
import yfinance as yf
import google.generativeai as genai

app = Flask(__name__)

# 配置 API Key
api_key = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

@app.route('/')
def home():
    return "伺服器運行中！請測試 /list 或 /predict"

# 【除錯專用】看看您的金鑰到底能用哪些模型
@app.route('/list')
def list_models():
    try:
        models = [m.name for m in genai.list_models()]
        return jsonify({"available_models": models})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/predict', methods=['GET'])
def predict():
    symbol = request.args.get('symbol', '2330.TW')
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period="1mo")
        if df.empty:
            return jsonify({"status": "error", "message": "找不到股票數據"}), 400
        
        latest_data = df.tail(10).to_string()

        # 這裡改用最保險的寫法： models/gemini-1.5-flash
        model = genai.GenerativeModel('models/gemini-1.5-flash')
        
        prompt = f"你是股票分析師，分析這段數據：\n{latest_data}"
        response = model.generate_content(prompt)
        
        return jsonify({
            "status": "success",
            "symbol": symbol,
            "analysis": response.text
        })
    except Exception as e:
        # 如果失敗，嘗試使用替代模型
        try:
            model_alt = genai.GenerativeModel('models/gemini-pro')
            response = model_alt.generate_content(f"分析股票: {symbol}")
            return jsonify({"status": "backup_success", "analysis": response.text})
        except:
            return jsonify({"status": "error", "details": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
