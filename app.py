from flask import Flask, render_template, request, jsonify
from gann_logic import analyze_stock
import json
import os

app = Flask(__name__)

# Load stock symbols on startup
def load_stock_data():
    try:
        with open('data/stock_list_20250527.csv', 'r') as f:
            import pandas as pd
            df = pd.read_csv(f)
            return [{'symbol': row['Symbol'], 'name': row['Company_Name']} 
                   for _, row in df.iterrows()]
    except Exception as e:
        print(f"Error loading stock data: {str(e)}")
        return []

STOCK_LIST = load_stock_data()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    stock_name = request.form.get('stock_name')
    if not stock_name:
        return render_template('index.html', error="Please enter a stock symbol")
    
    try:
        analysis = analyze_stock(stock_name)
        return render_template('result.html', analysis=analysis)
    except Exception as e:
        return render_template('index.html', error=f"Error analyzing stock: {str(e)}")

@app.route('/api/stocks')
def get_stocks():
    query = request.args.get('q', '').upper()
    if query:
        filtered_stocks = [
            stock for stock in STOCK_LIST 
            if query in stock['symbol'].upper() or query in stock['name'].upper()
        ][:10]  # Limit to 10 results
        return jsonify(filtered_stocks)
    return jsonify(STOCK_LIST[:100])  # Return first 100 stocks if no query

if __name__ == '__main__':
    app.run(debug=True, port=5000) 