import os
import json
import datetime
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS 
import yfinance as yf
import pandas as pd
import numpy as np
import google.generativeai as genai
from FinMind.data import DataLoader

app = Flask(__name__)
CORS(app) 

genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

@app.route('/')
def home():
    return "AI 戰情室大腦運轉中！(已搭載 PRO 級旗艦量化引擎)"

@app.route('/predict', methods=['GET'])
def predict():
    symbol = request.args.get('symbol', '2330.TW')
    try:
        # 1. 抓取技術面數據
        stock = yf.Ticker(symbol)
        df = stock.history(period="6mo")
        if df.empty:
            return jsonify({"status": "error", "message": f"無法獲取 {symbol} 數據。"}), 400

        # 計算 MACD 與 KD
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['DIF'] = df['EMA12'] - df['EMA26']
        df['MACD_Signal'] = df['DIF'].ewm(span=9, adjust=False).mean()
        df['OSC'] = df['DIF'] - df['MACD_Signal']

        df['9_high'] = df['High'].rolling(9).max()
        df['9_low'] = df['Low'].rolling(9).min()
        df['RSV'] = (df['Close'] - df['9_low']) / (df['9_high'] - df['9_low']) * 100
        df['RSV'] = df['RSV'].replace([np.inf, -np.inf], np.nan)
        
        rsv_list = df['RSV'].fillna(50).tolist()
        K, D = [], []
        prev_k, prev_d = 50, 50
        for rsv in rsv_list:
            curr_k = (2/3) * prev_k + (1/3) * rsv
            curr_d = (2/3) * prev_d + (1/3) * curr_k
            K.append(curr_k)
            D.append(curr_d)
            prev_k, prev_d = curr_k, curr_d
        df['K'] = K
        df['D'] = D
        
        df = df.fillna(0)
        df_chart = df.tail(60)
        
        chart_data, macd_data, kd_data = [], [], []
        for date, row in df_chart.iterrows():
            time_str = date.strftime('%Y-%m-%d')
            chart_data.append({"time": time_str, "open": row['Open'], "high": row['High'], "low": row['Low'], "close": row['Close']})
            macd_data.append({"time": time_str, "dif": row['DIF'], "signal": row['MACD_Signal'], "osc": row['OSC']})
            kd_data.append({"time": time_str, "k": row['K'], "d": row['D']})

        latest_data = df.tail(40).to_string()
        current_price = float(df['Close'].iloc[-1])

        try:
            company_name =
