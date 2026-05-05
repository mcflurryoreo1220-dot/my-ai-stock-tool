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
    "⚡ CPO 矽光子通訊": ["3163.TW", "3363.TW", "4979.TW", "6442.TW", "4908.TW"],
    "🏗️ 營造建材 (內需)": ["2504.TW", "2515.TW", "2520.TW", "1436.TW", "2501.TW"],
    "🔋 重電與綠能": ["1503.TW", "1504.TW", "1513.TW", "1514.TW", "1519.TW"]
}

RADAR_WATCHLIST = [s for group in SECTORS.values() for s in group] + ['2330.TW', '2317.TW']

@app.route('/')
def home():
    return "AI 戰情室大腦運轉中！(搭載跌停強制覆寫與高階防護網)"

def fetch_stock_basic(symbol):
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period="5d", interval="1d")
        if df.empty and symbol.endswith('.TW'):
            symbol = symbol.replace('.TW', '.TWO')
            stock = yf.Ticker(symbol)
            df = stock.history(period="5d", interval="1d")
        
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

@app.route('/radar', methods=['GET'])
def radar():
    matched = []
    # 雷達略過，保留核心路由
    return jsonify({"status": "success", "matches": matched})

@app.route('/predict', methods=['GET'])
def predict():
    symbol = request.args.get('symbol', '2330.TW')
    interval = request.args.get('interval', '1d')
    if interval not in ['1m', '5m', '15m', '60m', '1d']: interval = '1d'

    try:
        period = "5d" if interval in ['1m', '5m'] else ("1mo" if interval in ['15m', '60m'] else "6mo")
        stock = yf.Ticker(symbol)
        df = stock.history(period=period, interval=interval)
        if df.empty and symbol.endswith('.TW'):
            fallback_symbol = symbol.replace('.TW', '.TWO')
            stock = yf.Ticker(fallback_symbol)
            df = stock.history(period=period, interval=interval)
            symbol = fallback_symbol
        if df.empty: return jsonify({"status": "error", "message": "查無資料"}), 400

        df['MA5'] = df['Close'].rolling(window=5).mean(); df['MA20'] = df['Close'].rolling(window=20).mean(); df['MA60'] = df['Close'].rolling(window=60).mean()
        df['BB_std'] = df['Close'].rolling(window=20).std(); df['BB_upper'] = df['MA20'] + 2 * df['BB_std']; df['BB_lower'] = df['MA20'] - 2 * df['BB_std']
        df['Vol_MA5'] = df['Volume'].rolling(window=5).mean()
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean(); df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['DIF'] = df['EMA12'] - df['EMA26']; df['MACD_Signal'] = df['DIF'].ewm(span=9, adjust=False).mean(); df['OSC'] = df['DIF'] - df['MACD_Signal']
        df['9_high'] = df['High'].rolling(9).max(); df['9_low'] = df['Low'].rolling(9).min()
        df['RSV'] = ((df['Close'] - df['9_low']) / (df['9_high'] - df['9_low']) * 100).fillna(50)
        
        K, D = [], []; pk, pdv = 50, 50
        for rsv in df['RSV'].tolist(): ck = (2/3)*pk + (1/3)*rsv; cd = (2/3)*pdv + (1/3)*ck; K.append(ck); D.append(cd); pk, pdv = ck, cd
        df['K'], df['D'] = K, D
        df['Volume_Dir'] = np.sign(df['Close'].diff()).fillna(0)
        df['OBV'] = (df['Volume'] * df['Volume_Dir']).cumsum()

        df = df.fillna(0)
        chart_data, macd_data, kd_data = [], [], []
        for date, row in df.tail(80).iterrows():
            tv = date.strftime('%Y-%m-%d') if interval == '1d' else int(date.timestamp())
            chart_data.append({"time": tv, "open": round(row['Open'],2), "high": round(row['High'],2), "low": round(row['Low'],2), "close": round(row['Close'],2), "ma5": row['MA5'], "ma20": row['MA20'], "ma60": row['MA60'], "bb_upper": row['BB_upper'], "bb_lower": row['BB_lower']})
            macd_data.append({"time": tv, "dif": row['DIF'], "signal": row['MACD_Signal'], "osc": row['OSC']})
            kd_data.append({"time": tv, "k": row['K'], "d": row['D']})

        current_price = round(float(df['Close'].iloc[-1]), 2); pure_symbol = symbol.split('.')[0]
        fun_data = {"industry": "台股"}
        try:
            info = stock.info; display_name = STOCK_DICT.get(pure_symbol, info.get('shortName', pure_symbol))
            fun_data["industry"] = f"{info.get('sector', '')} {info.get('industry', '')}".strip() or "電子半導體"
        except: display_name = STOCK_DICT.get(pure_symbol, pure_symbol)

        last_row = df.iloc[-1]; prev_row = df.iloc[-2]
        change_amt = last_row['Close'] - prev_row['Close']
        change_pct = (change_amt / prev_row['Close']) * 100
        
        vol_data = {
            "today_vol": int(last_row['Volume']),
            "vol_ma5": int(last_row['Vol_MA5']),
            "price_change": round(change_amt, 2),
            "price_change_pct": round(change_pct, 2),
            "vol_change": int(last_row['Volume'] - prev_row['Volume'])
        }

        # === 【精準核心：跌停/漲停強制覆寫機制 (尚方寶劍)】 ===
        # 第一優先順位：極端行情判定，無視傳統量價公式
        if vol_data['price_change_pct'] <= -9.0:
            vol_data['status'] = "🚨 恐慌跌停"
            vol_data['desc'] = "流動性枯竭！跌停鎖死導致量縮假象，嚴禁接刀！"
            vol_data['color'] = "var(--green)" # 台股跌是綠色
        elif vol_data['price_change_pct'] >= 9.0:
            vol_data['status'] = "🔥 強勢漲停"
            vol_data['desc'] = "買盤極度強勢鎖死，籌碼完全掌控。"
            vol_data['color'] = "var(--red)"
        # 第二順位：常規量價背離判定
        elif vol_data['price_change'] > 0 and vol_data['vol_change'] >= 0:
            vol_data['status'] = "價漲量增"
            vol_data['desc'] = "健康上漲格局，買盤推升。"
            vol_data['color'] = "var(--red)"
        elif vol_data['price_change'] > 0 and vol_data['vol_change'] < 0:
            vol_data['status'] = "量價背離 (漲)"
            vol_data['desc'] = "價漲但量縮，追高意願降低，留意反轉。"
            vol_data['color'] = "var(--orange)"
        elif vol_data['price_change'] <= 0 and vol_data['vol_change'] > 0:
            vol_data['status'] = "價跌量增"
            vol_data['desc'] = "賣壓出籠，主力疑似出貨，請提高警覺！"
            vol_data['color'] = "var(--green)"
        else:
            vol_data['status'] = "價跌量縮"
            vol_data['desc'] = "量縮整理，觀察下檔支撐是否守穩。"
            vol_data['color'] = "var(--text-muted)"

        chip_table_data = []
        net_foreign_5d = 0
        if interval == '1d':
            try:
                dl = DataLoader()
                df_chips = dl.taiwan_stock_institutional_investors(stock_id=pure_symbol, start_date=(datetime.datetime.now() - datetime.timedelta(days=20)).strftime('%Y-%m-%d'))
                if not df_chips.empty:
                    df_chips['net_buy'] = df_chips['buy'] - df_chips['sell']
                    df_chips['name'] = df_chips['name'].replace({'外資及陸資(不含外資自營商)': '外資', '外資及陸資': '外資', '自營商(自行買賣)': '自營', '自營商(避險)': '自營', '自營商': '自營'})
                    pv = df_chips.groupby(['date', 'name'])['net_buy'].sum().unstack(fill_value=0).reset_index()
                    for col in ['外資', '投信', '自營']:
                        if col not in pv.columns: pv[col] = 0
                    pv['合計'] = pv['外資'] + pv['投信'] + pv['自營']
                    net_foreign_5d = pv.tail(5)['外資'].sum()
                    for _, r in pv.tail(10).iloc[::-1].iterrows():
                        chip_table_data.append({"date": str(r['date'])[5:], "foreign": round(r['外資']/1000,1), "trust": round(r['投信']/1000,1), "dealer": round(r['自營']/1000,1), "total": round(r['合計']/1000,1)})
            except: pass

        # === 【精準核心：動態警示系統】 ===
        warning_box = {"active": False, "title": "安全", "msg": "目前無明顯出貨跡象", "level": "safe"}
        if vol_data['price_change_pct'] <= -9.0:
            warning_box = {"active": True, "title": "🚨 跌停警報", "msg": "極端空頭鎖死，無量下殺，請啟動絕對防守機制，切勿摸底！", "level": "danger"}
        elif vol_data['status'] == "價跌量增" or (vol_data['status'] == "量價背離 (漲)" and net_foreign_5d < 0):
            warning_box = {"active": True, "title": "⚠️ 主力警示", "msg": "量價結構轉弱，疑似主力逢高調節，請嚴格控管資金部位！", "level": "warning"}

        fallback_signal = "區間震盪"
        if vol_data['price_change_pct'] <= -9.0: fallback_signal = "極度空頭 (跌停)"
        elif vol_data['price_change_pct'] >= 9.0: fallback_signal = "極度多頭 (漲停)"
        elif last_row['K'] > last_row['D'] and last_row['Close'] > last_row['MA20']: fallback_signal = "多頭格局"
        elif last_row['K'] < last_row['D'] and last_row['Close'] < last_row['MA20']: fallback_signal = "空頭弱勢"

        ai_data = {
            "op_short": "空頭宣洩中，嚴禁接刀" if fallback_signal=="極度空頭 (跌停)" else "均線附近分批佈局", 
            "vol_price_div": vol_data['status'], 
            "entry_winrate": "0% (跌停風險)" if fallback_signal=="極度空頭 (跌停)" else "待計算",
            "mid_long_view": "基本面與籌碼拉扯中", 
            "vol_analysis": vol_data['desc'], 
            "exit_warning": warning_box['title'],
            "key_levels": f"支撐 {round(last_row['BB_lower'],2)} / 壓力 {round(last_row['MA20'],2)}", 
            "risk_reminder": warning_box['msg'],
            "prob_up": 10 if fallback_signal=="極度空頭 (跌停)" else 33, 
            "prob_down": 80 if fallback_signal=="極度空頭 (跌停)" else 33, 
            "prob_flat": 10 if fallback_signal=="極度空頭 (跌停)" else 34, 
            "signal": fallback_signal, "pressure": str(round(last_row['BB_upper'], 2)), "support": str(round(last_row['BB_lower'], 2)), "stop_loss": str(round(last_row['MA20'], 2)),
            "industry_desc": fun_data["industry"]
        }
        
        prompt = (
            f"請扮演避險基金策略主管。分析股票 {display_name} ({pure_symbol})。\n"
            f"重要情報：今日漲跌幅 {vol_data['price_change_pct']}%\n"
            f"若漲跌幅 <= -9%，請務必將操作建議設為極度保守(不可接刀)。\n"
            f"務必只輸出純 JSON，格式如下：\n"
            f"{{\n"
            f"  \"op_short\": \"1.操作建議(短線)(15字內)\", \"vol_price_div\": \"2.量價結構(15字內)\", \"entry_winrate\": \"3.短線勝率(15字內)\",\n"
            f"  \"mid_long_view\": \"4.中長線看法(15字內)\", \"vol_analysis\": \"5.成交量分析(15字內)\", \"exit_warning\": \"6.主力出貨警示(10字內)\",\n"
            f"  \"key_levels\": \"7.關鍵價位(15字內)\", \"risk_reminder\": \"8.風險提醒(15字內)\",\n"
            f"  \"prob_up\": 30, \"prob_down\": 40, \"prob_flat\": 30,\n"
            f"  \"signal\": \"多/空/震盪\", \"pressure\": \"壓力價\", \"support\": \"支撐價\", \"stop_loss\": \"停損價\"\n"
            f"}}\n"
        )
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0.1))
            match = re.search(r'\{[\s\S]*\}', response.text)
            if match:
                parsed = json.loads(match.group(0))
                for k, v in parsed.items():
                    if k in ai_data: ai_data[k] = v
        except: pass

        return jsonify({
            "status": "success", "symbol": symbol, "current_price": current_price, "interval": interval,
            "chart_data": chart_data, "macd_data": macd_data, "kd_data": kd_data, 
            "chip_table": chip_table_data, "fundamental": fun_data, "ai_analysis": ai_data,
            "volume_data": vol_data, "warning_box": warning_box
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))
