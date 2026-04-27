import os
import json
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
        df = stock.history(period="3mo")
        if df.empty:
            return jsonify({"status": "error", "message": "找不到股票數據"}), 400
        
        # 打包 K 線圖表所需的資料
        chart_data = []
        for date, row in df.iterrows():
            chart_data.append({
                "time": date.strftime('%Y-%m-%d'),
                "open": float(row['Open']),
                "high": float(row['High']),
                "low": float(row['Low']),
                "close": float(row['Close'])
            })

        latest_data = df.tail(40).to_string()
        current_price = float(df['Close'].iloc[-1])

        # 抓取真實公司名稱防幻覺
        try:
            company_name = stock.info.get('shortName', '')
            display_name = f"{company_name} ({symbol})" if company_name else symbol
        except:
            display_name = symbol

        # 2. 智慧篩選模型
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        flash_15_models = [m for m in available_models if '1.5-flash' in m]
        target_model = flash_15_models[0] if flash_15_models else (available_models[0] if available_models else 'gemini-1.5-flash')

        model = genai.GenerativeModel(target_model)
        
        # 3. 【關鍵升級】：強制 AI 輸出 JSON 格式的結構化數據
        prompt = (
            f"你是一位擁有 20 年經驗的台股操盤手。請針對 {display_name} 的數據進行分析。\n"
            f"【極重要指示】：\n"
            f"你必須以純 JSON 格式輸出結果，絕對不要包含任何 Markdown 標記 (例如 ```json)。\n"
            f"JSON 的 Key 必須嚴格包含以下五個項目：\n"
            f"1. \"trend\" (字串，判斷目前趨勢，如：短線偏強、震盪整理、破底危機)\n"
            f"2. \"pressure\" (數字，近期的關鍵壓力價位)\n"
            f"3. \"support\" (數字，近期的關鍵支撐價位)\n"
            f"4. \"summary\" (字串，100字以內的技術面與K線型態總結)\n"
            f"5. \"action\" (字串，具體的操作建議，如：空手觀望、逢回找買點、分批停利)\n\n"
            f"數據：\n{latest_data}"
        )
        
        response = model.generate_content(prompt)
        
        # 4. 解析 AI 回傳的 JSON 資料
        try:
            # 清除 AI 有時會雞婆加上的 markdown 標籤
            raw_text = response.text.replace("```json", "").replace("```", "").strip()
            ai_data = json.loads(raw_text)
        except Exception as parse_err:
            # 萬一 AI 格式寫錯的防呆機制
            ai_data = {
                "trend": "解析失敗",
                "pressure": 0,
                "support": 0,
                "summary": f"AI 回傳格式異常，請再試一次。原始文字: {response.text[:50]}...",
                "action": "暫停操作"
            }
        
        return jsonify({
            "status": "success",
            "symbol": symbol,
            "current_price": current_price,
            "chart_data": chart_data,
            "ai_analysis": ai_data, # 將結構化的資料傳給網頁
            "using_model": target_model
        })
    except Exception as e:
        return jsonify({"status": "error", "message": f"系統錯誤: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
