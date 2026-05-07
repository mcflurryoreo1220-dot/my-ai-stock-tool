import os
import json
import datetime
import traceback
import re
import concurrent.futures
from flask import Flask, request, jsonify
from flask_cors import CORS 
import yfinance as yf
import pandas as pd
import numpy as np
import google.generativeai as genai
from FinMind.data import DataLoader
import pytz

app = Flask(__name__)
CORS(app) 

api_key = os.environ.get("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

STOCK_DICT = {
    "2382": "廣達", "3231": "緯創", "2376": "技嘉", "3324": "雙鴻", "3017": "奇鋐",
    "3661": "世芯-KY", "3131": "弘塑", "6187": "萬潤", "6683": "雍智科技", "3583": "辛耘",
    "3163": "波若威", "3363": "上詮", "4979": "華星光", "6442": "光聖", "4908": "前鼎",
    "2504": "國產", "2515": "中工", "2520": "冠德", "1436": "華友聯", "2501": "國建",
    "1503": "士電", "1504": "東元", "1513": "中興電", "1514": "亞力", "1519": "華城",
    "2330": "台積電", "2317": "鴻海", "2454": "聯發科", "2301": "光寶科", "2441": "超豐",
    "6805": "富世達"
}

SECTORS = {
    "🔥 AI 伺服器 & 散熱": ["2382.TW", "3231.TW", "2376.TW", "3324.TW", "3017.TW", "6805.TW"],
    "🚀 CoWoS 先進封裝": ["3661.TW", "3131.TW", "6187.TW", "6683.TW", "3583.TW"],
    "🏗️ 營造建材 (內需)": ["2504.TW", "2515.TW", "2520.TW", "1436.TW", "2501.TW"],
    "🔋 重電與綠能": ["1503.TW", "1504.TW", "1513.TW", "1514.TW", "1519.TW"]
}

@app.route('/')
def home():
    return "AI 戰情室大腦運轉中！(搭載 Alpha 對比與深度連結優化)"

def fetch_stock_basic(symbol):
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period="5d", interval="1d")
        if df.empty and symbol.endswith('.TW'):
            symbol = symbol.replace('.TW', '.TWO')
            stock = yf.Ticker(symbol); df = stock.history(period="5d", interval="1d")
        if len(df) >= 2:
            curr = df.iloc[-1]['Close']; prev = df.iloc[-2]['Close']
            change_pct = ((curr - prev) / prev) * 100
            pure_sym = symbol.split('.')[0]
            name = STOCK_DICT.get(pure_sym, stock.info.get('shortName', pure_sym))
            return {"symbol": pure_sym, "name": name, "price": round(curr, 2), "change": round(change_pct, 2)}
    except: pass
    return None

@app.route('/sectors', methods=['GET'])
def get_sectors():
    sector_results = {}
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            for sector_name, symbols in SECTORS.items():
                results = list(executor.map(fetch_stock_basic, symbols))
                valid_results = [r for r in results if r]
                valid_results.sort(key=lambda x: x['change'], reverse=True)
                sector_results[sector_name] = valid_results
        return jsonify({"status": "success", "data": sector_results})
    except: return jsonify({"status": "error"}), 500

@app.route('/predict', methods=['GET'])
def predict():
    symbol = request.args.get('symbol', '2330.TW')
    interval = request.args.get('interval', '1d')
    if interval not in ['1m', '5m', '15m', '60m', '1d']: interval = '1d'

    try:
        period = "5d" if interval in ['1m', '5m'] else ("1mo" if interval in ['15m', '60m'] else "6mo")
        stock = yf.Ticker(symbol); df = stock.history(period=period, interval=interval)
        if df.empty and symbol.endswith('.TW'):
            fallback_symbol = symbol.replace('.TW', '.TWO')
            stock = yf.Ticker(fallback_symbol); df = stock.history(period=period, interval=interval)
            symbol = fallback_symbol
        if df.empty: return jsonify({"status": "error", "message": "查無資料"}), 400

        # 技術指標計算
        df['MA20'] = df['Close'].rolling(window=20).mean(); df['BB_std'] = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['MA20'] + 2 * df['BB_std']; df['BB_lower'] = df['MA20'] - 2 * df['BB_std']
        df['Vol_MA5'] = df['Volume'].rolling(window=5).mean()
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean(); df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['OSC'] = (df['EMA12'] - df['EMA26']) - (df['EMA12'] - df['EMA26']).ewm(span=9, adjust=False).mean()
        df['9_high'] = df['High'].rolling(9).max(); df['9_low'] = df['Low'].rolling(9).min()
        df['RSV'] = ((df['Close'] - df['9_low']) / (df['9_high'] - df['9_low']) * 100).fillna(50)
        K, D, pk, pdv = [], [], 50, 50
        for rsv in df['RSV'].tolist(): ck = (2/3)*pk + (1/3)*rsv; cd = (2/3)*pdv + (1/3)*ck; K.append(ck); D.append(cd); pk, pdv = ck, cd
        df['K'], df['D'] = K, D

        chart_data, macd_data, kd_data = [], [], []
        for date, row in df.tail(80).iterrows():
            tv = date.strftime('%Y-%m-%d') if interval == '1d' else int(date.timestamp())
            chart_data.append({"time": tv, "open": round(row['Open'],2), "high": round(row['High'],2), "low": round(row['Low'],2), "close": round(row['Close'],2), "ma20": row['MA20'], "bb_upper": row['BB_upper'], "bb_lower": row['BB_lower']})
            kd_data.append({"time": tv, "k": row['K'], "d": row['D']})

        current_price = round(float(df['Close'].iloc[-1]), 2); pure_symbol = symbol.split('.')[0]
        
        # === 抓取大盤對比 Alpha ===
        try:
            taiex = yf.Ticker("^TWII").history(period="5d", interval="1d")
            taiex_change = ((taiex.iloc[-1]['Close'] - taiex.iloc[-2]['Close']) / taiex.iloc[-2]['Close']) * 100
        except: taiex_change = 0

        last_row = df.iloc[-1]; prev_row = df.iloc[-2]
        change_pct = ((last_row['Close'] - prev_row['Close']) / prev_row['Close']) * 100
        alpha_strength = "強於大盤" if change_pct > taiex_change else "弱於大盤"

        # 警示與量價
        vol_status = "價漲量增" if change_pct > 0 and last_row['Volume'] > prev_row['Volume'] else "價漲量縮"
        if change_pct < 0 and last_row['Volume'] > prev_row['Volume']: vol_status = "價跌量增"
        
        warning_box = {"active": False, "title": "安全", "msg": "無明顯異常"}
        if change_pct <= -9.0: warning_box = {"active": True, "title": "🚨 跌停警報", "msg": "恐慌宣洩，切勿摸底"}
        elif vol_status == "價跌量增" or (change_pct > 0 and last_row['Volume'] < prev_row['Volume']):
            warning_box = {"active": True, "title": "⚠️ 主力警示", "msg": "量價背離或出貨跡象"}

        # AI 戰術指揮
        prompt = (
            f"你是一位避險基金分析師。分析股票 {pure_symbol}。\n"
            f"今日漲跌 {change_pct:.2f}%，大盤漲跌 {taiex_change:.2f}%。\n"
            f"務必只輸出純 JSON：\n"
            f"{{\n"
            f"  \"op_short\": \"短線操作建議(15字)\", \"vol_price_div\": \"{vol_status}\", \"entry_winrate\": \"具體勝率與理由\",\n"
            f"  \"mid_long_view\": \"中長線展望\", \"vol_analysis\": \"今日成交量特徵\", \"exit_warning\": \"{warning_box['title']}\",\n"
            f"  \"key_levels\": \"關鍵壓力支撐\", \"risk_reminder\": \"{warning_box['msg']}\",\n"
            f"  \"prob_up\": 40, \"prob_down\": 30, \"prob_flat\": 30,\n"
            f"  \"moat_score\": \"護城河評分\", \"market_narrative\": \"目前市場炒作敘事\", \"bull_bear\": \"牛熊預測\"\n"
            f"}}\n"
        )
        
        ai_data = {"op_short": "觀察中", "vol_price_div": vol_status, "entry_winrate": "--", "prob_up": 33, "prob_down": 33, "prob_flat": 34, "moat_score": "7"}
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0.2))
            match = re.search(r'\{[\s\S]*\}', response.text)
            if match: ai_data.update(json.loads(match.group(0)))
        except: pass

        return jsonify({
            "status": "success", "symbol": symbol, "current_price": current_price,
            "chart_data": chart_data, "kd_data": kd_data, "ai_analysis": ai_data,
            "alpha": {"taiex": round(taiex_change, 2), "strength": alpha_strength},
            "warning": warning_box
        })
    except Exception as e: return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))
