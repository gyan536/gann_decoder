from flask import Flask, render_template, request, jsonify
from gann_logic import analyze_stock
from ml_market_analysis import MLMarketAnalyzer
import yfinance as yf
from datetime import datetime, timedelta
import json
import os
import pandas as pd
import plotly
import numpy as np
import re
from gann_analysis import GannAnalysis
import logging
from get_all_stocks import stock_manager

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Common stock symbol mappings
SYMBOL_MAPPINGS = {
    # Indian Banks
    'HDFC BANK': 'HDFCBANK.NS',
    'ICICI BANK': 'ICICIBANK.NS',
    'STATE BANK OF INDIA': 'SBIN.NS',
    'SBI': 'SBIN.NS',
    'AXIS BANK': 'AXISBANK.NS',
    'KOTAK MAHINDRA BANK': 'KOTAKBANK.NS',
    'KOTAK BANK': 'KOTAKBANK.NS',
    
    # Indian IT
    'TCS': 'TCS.NS',
    'TATA CONSULTANCY': 'TCS.NS',
    'INFOSYS': 'INFY.NS',
    'WIPRO': 'WIPRO.NS',
    'HCL TECH': 'HCLTECH.NS',
    'HCL': 'HCLTECH.NS',
    
    # Indian Conglomerates
    'RELIANCE': 'RELIANCE.NS',
    'RELIANCE INDUSTRIES': 'RELIANCE.NS',
    'TATA MOTORS': 'TATAMOTORS.NS',
    'TATA STEEL': 'TATASTEEL.NS',
    'L&T': 'LT.NS',
    'LARSEN': 'LT.NS',
    'LARSEN & TOUBRO': 'LT.NS',
    
    # US Tech
    'APPLE': 'AAPL',
    'MICROSOFT': 'MSFT',
    'GOOGLE': 'GOOGL',
    'ALPHABET': 'GOOGL',
    'AMAZON': 'AMZN',
    'META': 'META',
    'FACEBOOK': 'META',
    'NETFLIX': 'NFLX',
    'TESLA': 'TSLA'
}

def clean_symbol(symbol: str) -> str:
    """Clean and standardize symbol format"""
    # Remove common suffixes and clean
    clean = symbol.upper().strip()
    for suffix in [' LTD', ' LIMITED', ' LTD.', ' CORP', ' CORPORATION', ' INC', ' INC.']:
        clean = clean.replace(suffix.upper(), '')
    return clean

def map_symbol(symbol: str) -> str:
    """Map any stock symbol/name to its correct trading symbol"""
    try:
        original_symbol = symbol
        clean = clean_symbol(symbol)
        
        # Check direct mapping first
        if clean in SYMBOL_MAPPINGS:
            mapped = SYMBOL_MAPPINGS[clean]
            # Verify the mapped symbol works
            try:
                ticker = yf.Ticker(mapped)
                info = ticker.info
                if info and 'regularMarketPrice' in info and info['regularMarketPrice'] is not None:
                    return mapped
            except:
                pass

        # If it's already in correct format, verify it works
        if re.match(r'^[A-Z\d]+\.(NS|BO)$', symbol) or re.match(r'^[A-Z\d]+$', symbol):
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                if info and 'regularMarketPrice' in info and info['regularMarketPrice'] is not None:
                    return symbol
            except:
                pass

        # Try different formats
        test_symbols = []
        
        # Add original clean symbol
        test_symbols.append(clean)
        
        # Try NSE format
        test_symbols.append(f"{clean}.NS")
        
        # Try BSE format
        test_symbols.append(f"{clean}.BO")
        
        # Try without exchange suffix
        if clean.endswith('.NS') or clean.endswith('.BO'):
            test_symbols.append(clean[:-3])
        
        # Try fuzzy matches from mappings
        for known_name, known_symbol in SYMBOL_MAPPINGS.items():
            if (clean in known_name.upper() or 
                known_name.upper() in clean or 
                clean in known_symbol.upper()):
                test_symbols.append(known_symbol)

        # Test each symbol
        for test_symbol in test_symbols:
            try:
                ticker = yf.Ticker(test_symbol)
                info = ticker.info
                if info and 'regularMarketPrice' in info and info['regularMarketPrice'] is not None:
                    return test_symbol
            except:
                continue

        # If nothing worked, raise a helpful error
        suggestions = []
        if clean.endswith('.NS') or clean.endswith('.BO'):
            suggestions.append(f"Try without exchange suffix: {clean[:-3]}")
        else:
            suggestions.append(f"Try with exchange suffix: {clean}.NS (for NSE) or {clean}.BO (for BSE)")
        
        # Add similar matches from mappings
        similar_matches = []
        for name, symbol in SYMBOL_MAPPINGS.items():
            if any(word in name.upper().split() for word in clean.split()):
                similar_matches.append(f"{name} ({symbol})")
        
        if similar_matches:
            suggestions.append("Similar stocks: " + ", ".join(similar_matches[:3]))
        
        raise ValueError(
            f"Could not find valid symbol for '{original_symbol}'. \n" + 
            "\n".join(suggestions)
        )

    except Exception as e:
        if "Could not find valid symbol" in str(e):
            raise
        raise ValueError(f"Error validating symbol '{original_symbol}': {str(e)}")

# Load stock symbols on startup
def load_stock_data():
    try:
        with open('data/stock_list_20250527.csv', 'r') as f:
            df = pd.read_csv(f)
            return [{'symbol': row['Symbol'], 'name': row['Company_Name']} 
                   for _, row in df.iterrows()]
    except Exception as e:
        print(f"Error loading stock data: {str(e)}")
        return []

STOCK_LIST = load_stock_data()

# Cache for ML analyzers to avoid retraining
ml_analyzers = {}

def get_ml_analyzer(symbol: str, force_refresh: bool = False) -> MLMarketAnalyzer:
    """Get or create an ML analyzer for a given symbol"""
    if symbol not in ml_analyzers or force_refresh:
        try:
            # Map the symbol to correct format
            mapped_symbol = map_symbol(symbol)
            print(f"Mapped '{symbol}' to '{mapped_symbol}'")
            
            # Download 1 year of historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            df = yf.download(mapped_symbol, start=start_date, end=end_date)
            
            # Check if we got any data
            if df.empty:
                raise ValueError(f"No data available for symbol {symbol} (mapped to {mapped_symbol})")
            
            # Check if we have enough data points
            if len(df) < 50:  # Minimum required data points
                raise ValueError(f"Insufficient data for symbol {mapped_symbol}. Need at least 50 data points.")
            
            # Create and train analyzer
            analyzer = MLMarketAnalyzer(df)
            analyzer.train_models()
            ml_analyzers[symbol] = analyzer
            
        except Exception as e:
            raise ValueError(str(e))
    
    return ml_analyzers[symbol]

@app.route('/')
def index():
    return render_template('index.html')

def fetch_stock_data(symbol):
    """Fetch stock data with proper headers and error handling"""
    try:
        # Create a yfinance Ticker object
        stock = yf.Ticker(symbol)
        
        # Try to get 1 year of data
        hist = stock.history(period='1y')
        
        if hist.empty:
            raise ValueError(f"No data available for symbol {symbol}")
        
        # Verify we have the required columns
        required_columns = ['Open', 'High', 'Low', 'Close']
        if not all(col in hist.columns for col in required_columns):
            raise ValueError(f"Incomplete data for symbol {symbol}")
        
        # Verify we have recent data (last 5 days) with proper timezone handling
        if len(hist) > 0:
            last_date = hist.index[-1]
            # Convert last_date to datetime if it's not already
            if not isinstance(last_date, datetime):
                last_date = pd.to_datetime(last_date)
            # Remove timezone information for comparison
            last_date = last_date.replace(tzinfo=None)
            current_date = datetime.now()
            
            days_difference = (current_date - last_date).days
            if days_difference > 5:
                raise ValueError(f"Data for {symbol} is not up to date. Last available date: {last_date.strftime('%Y-%m-%d')}")
        else:
            raise ValueError(f"No historical data available for {symbol}")
        
        return hist, stock
        
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
        raise ValueError(f"Unable to fetch data for {symbol}: {str(e)}")

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        stock_symbol = request.form.get('stock_name', '').strip().upper()
        if not stock_symbol:
            return render_template('index.html', error='Please enter a stock symbol')

        # Validate the stock symbol
        validation = stock_manager.validate_symbol(stock_symbol)
        if not validation['valid']:
            if 'suggestions' in validation and validation['suggestions']:
                suggestions = [f"{s['symbol']} ({s['name']})" for s in validation['suggestions']]
                error_msg = f"{validation['message']}\nSuggestions: {', '.join(suggestions)}"
            else:
                error_msg = validation['message']
            return render_template('index.html', error=error_msg)

        try:
            # Use the validated symbol
            stock_symbol = validation['symbol']
            
            # Fetch stock data with improved error handling
            try:
                hist, stock = fetch_stock_data(stock_symbol)
            except ValueError as e:
                return render_template('index.html', error=str(e))
            except Exception as e:
                logger.error(f"Unexpected error fetching data for {stock_symbol}: {str(e)}")
                return render_template('index.html', 
                    error=f"Unable to fetch data for {stock_symbol}. The service might be temporarily unavailable. Please try again later.")

            # Calculate Gann analysis
            try:
                analysis = calculate_gann_analysis(hist)
            except Exception as e:
                logger.error(f"Error in Gann analysis for {stock_symbol}: {str(e)}")
                return render_template('index.html', error=f"Error calculating analysis for {stock_symbol}: {str(e)}")
            
            # Add the stock symbol and info to the analysis
            analysis['stock'] = stock_symbol
            
            # Try to get company name and additional info
            try:
                info = stock.info
                if info and isinstance(info, dict):
                    analysis['company_name'] = info.get('longName', '') or info.get('shortName', '') or stock_symbol
                    # Add more relevant info if available
                    analysis['sector'] = info.get('sector', '')
                    analysis['industry'] = info.get('industry', '')
                    analysis['market_cap'] = info.get('marketCap', 0)
                else:
                    analysis['company_name'] = stock_symbol
            except Exception as e:
                logger.error(f"Error fetching company info for {stock_symbol}: {str(e)}")
                analysis['company_name'] = stock_symbol

            # Format numbers in analysis with error checking
            try:
                analysis['current_price'] = float(analysis['current_price'])
                analysis['resistance_levels'] = [float(x) for x in analysis['resistance_levels']]
                analysis['support_levels'] = [float(x) for x in analysis['support_levels']]
                analysis['confidence'] = float(analysis['confidence'])
                analysis['time_cycles'] = {k: float(v) for k, v in analysis['time_cycles'].items()}
                if 'market_cap' in analysis:
                    analysis['market_cap'] = float(analysis['market_cap'])
            except Exception as e:
                logger.error(f"Error formatting analysis values for {stock_symbol}: {str(e)}")
                return render_template('index.html', error=f"Error processing analysis results for {stock_symbol}")
            
            # Ensure all required fields are present
            required_fields = ['current_price', 'trend', 'momentum', 'resistance_levels', 
                             'support_levels', 'time_cycles', 'recommendation', 'confidence']
            if not all(field in analysis for field in required_fields):
                return render_template('index.html', error=f"Incomplete analysis results for {stock_symbol}")
            
            return render_template(
                'result.html',
                analysis=analysis,
                original_symbol=request.form.get('stock_name')
            )
            
        except Exception as e:
            logger.error(f"Analysis error for {stock_symbol}: {str(e)}")
            return render_template('index.html', 
                                error=f"Error analyzing {stock_symbol}: {str(e)}")
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return render_template('index.html', 
                             error="An unexpected error occurred. Please try again.")

def calculate_gann_analysis(hist):
    """Calculate Gann analysis for the given historical data"""
    try:
        # Validate input data
        if hist is None or hist.empty:
            raise ValueError("No historical data provided")
        
        if 'Close' not in hist.columns:
            raise ValueError("Historical data missing 'Close' prices")
        
        # Get the last valid close price
        current_price = hist['Close'].dropna().iloc[-1]
        if not isinstance(current_price, (int, float)):
            current_price = float(current_price)
        
        # Calculate trend with error handling
        try:
            short_ma = hist['Close'].rolling(window=20).mean().dropna().iloc[-1]
            long_ma = hist['Close'].rolling(window=50).mean().dropna().iloc[-1]
            if not isinstance(short_ma, (int, float)):
                short_ma = float(short_ma)
            if not isinstance(long_ma, (int, float)):
                long_ma = float(long_ma)
            trend = 'BULLISH' if short_ma > long_ma else 'BEARISH'
        except Exception as e:
            logger.error(f"Error calculating trend: {str(e)}")
            trend = 'NEUTRAL'  # Default if calculation fails
        
        # Calculate momentum with error handling
        try:
            pct_change = hist['Close'].pct_change(20).dropna().iloc[-1]
            if not isinstance(pct_change, (int, float)):
                pct_change = float(pct_change)
            momentum = 'BULLISH' if pct_change > 0 else 'BEARISH'
        except Exception as e:
            logger.error(f"Error calculating momentum: {str(e)}")
            momentum = 'NEUTRAL'  # Default if calculation fails
        
        # Calculate Gann levels with error handling
        try:
            sqrt_price = np.sqrt(current_price)
            levels = np.square(np.arange(sqrt_price - 5, sqrt_price + 5, 0.5))
            
            resistance_levels = sorted([float(l) for l in levels if l > current_price])[:3]
            support_levels = sorted([float(l) for l in levels if l < current_price], reverse=True)[:3]
            
            # Ensure we have at least one level
            if not resistance_levels:
                resistance_levels = [float(current_price * 1.01)]
            if not support_levels:
                support_levels = [float(current_price * 0.99)]
        except Exception as e:
            logger.error(f"Error calculating Gann levels: {str(e)}")
            # Provide default levels if calculation fails
            resistance_levels = [float(current_price * 1.01)]
            support_levels = [float(current_price * 0.99)]
        
        # Calculate time cycles
        time_cycles = calculate_time_cycles(hist)
        
        # Generate recommendation based on trend and momentum
        if trend == 'BULLISH' and momentum == 'BULLISH':
            recommendation = 'BUY'
            confidence = 0.8
        elif trend == 'BEARISH' and momentum == 'BEARISH':
            recommendation = 'SELL'
            confidence = 0.8
        else:
            recommendation = 'HOLD'
            confidence = 0.5
        
        # Calculate suggested time (21 days from now)
        suggested_time = (datetime.now() + timedelta(days=21)).strftime('%Y-%m-%d')
        
        # Create and validate the analysis dictionary
        analysis = {
            'current_price': float(current_price),
            'trend': trend,
            'momentum': momentum,
            'resistance_levels': [float(x) for x in resistance_levels],
            'support_levels': [float(x) for x in support_levels],
            'time_cycles': {k: float(v) for k, v in time_cycles.items()},
            'recommendation': recommendation,
            'confidence': float(confidence),
            'suggested_time': suggested_time,
            'explanation': f"Based on {trend.lower()} trend and {momentum.lower()} momentum"
        }
        
        # Validate all values are properly formatted
        for key, value in analysis.items():
            if key in ['current_price', 'confidence']:
                if not isinstance(value, float):
                    raise ValueError(f"Invalid type for {key}: expected float, got {type(value)}")
            elif key in ['resistance_levels', 'support_levels']:
                if not isinstance(value, list) or not all(isinstance(x, float) for x in value):
                    raise ValueError(f"Invalid type for {key}: expected list of floats")
            elif key == 'time_cycles':
                if not isinstance(value, dict) or not all(isinstance(v, float) for v in value.values()):
                    raise ValueError(f"Invalid type for {key}: expected dict with float values")
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error in Gann analysis calculation: {str(e)}")
        raise ValueError(f"Failed to calculate analysis: {str(e)}")

def calculate_time_cycles(hist):
    """Calculate time cycle correlations"""
    try:
        if hist is None or hist.empty:
            raise ValueError("No historical data provided")
        
        cycles = {
            '90': 90,
            '144': 144,
            '180': 180,
            '270': 270,
            '360': 360
        }
        
        correlations = {}
        for name, period in cycles.items():
            try:
                if len(hist) >= period:
                    correlation = hist['Close'].autocorr(lag=period)
                    correlations[name] = float(correlation if not np.isnan(correlation) else 0)
                else:
                    correlations[name] = 0.0
            except Exception as e:
                logger.error(f"Error calculating correlation for period {period}: {str(e)}")
                correlations[name] = 0.0
        
        return correlations
        
    except Exception as e:
        logger.error(f"Error calculating time cycles: {str(e)}")
        # Return default values if calculation fails
        return {'90': 0.0, '144': 0.0, '180': 0.0, '270': 0.0, '360': 0.0}

@app.route('/api/stocks')
def search_stocks():
    query = request.args.get('q', '')
    if not query or len(query) < 2:
        return jsonify([])
    
    matches = stock_manager.search_symbol(query)
    return jsonify([{
        'symbol': m['symbol'],
        'name': m['name']
    } for m in matches])

@app.route('/api/refresh-analysis', methods=['POST'])
def refresh_analysis():
    """Force refresh analysis for a symbol"""
    stock_name = request.json.get('symbol')
    if not stock_name:
        return jsonify({'error': 'Symbol required'}), 400
    
    try:
        mapped_symbol = map_symbol(stock_name)
        analyzer = get_ml_analyzer(mapped_symbol, force_refresh=True)
        market_state = analyzer.analyze_market_state()
        return jsonify({
            'status': 'success',
            'market_state': market_state,
            'mapped_symbol': mapped_symbol
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def create_analysis_plot(df, waves, next_wave):
    """Create interactive plot for analysis results"""
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=('Price Action & Waves', 'Confidence & Risk'))
    
    # Plot candlesticks
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price Action'
        ),
        row=1, col=1
    )
    
    # Plot waves
    for wave in waves:
        color = 'green' if wave.direction.value == 'UP' else 'red'
        fig.add_trace(
            go.Scatter(
                x=[wave.start_time, wave.end_time],
                y=[wave.start_price, wave.end_price],
                mode='lines',
                line=dict(color=color, width=2),
                name=f'{wave.direction.value} Wave ({wave.confidence:.2f})'
            ),
            row=1, col=1
        )
    
    # Plot confidence levels
    if waves:  # Only plot if we have waves
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=[wave.confidence for wave in waves] * len(df),
                name='Wave Confidence',
                line=dict(color='blue')
            ),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        title='ML Market Analysis',
        yaxis_title='Price',
        yaxis2_title='Confidence/Risk',
        xaxis_rangeslider_visible=False,
        height=800
    )
    
    return fig

def verify_yfinance_connectivity():
    """Verify yfinance installation and connectivity"""
    try:
        # Try to fetch data for a well-known stock
        test_symbol = 'AAPL'  # Apple Inc. is usually reliable
        
        stock = yf.Ticker(test_symbol)
        hist = stock.history(period='1d')
        
        if hist.empty:
            logger.error("yfinance connectivity test failed: No data received")
            return False
            
        # Verify data with timezone-aware comparison
        if len(hist) > 0:
            last_date = hist.index[-1]
            if not isinstance(last_date, datetime):
                last_date = pd.to_datetime(last_date)
            last_date = last_date.replace(tzinfo=None)
            current_date = datetime.now()
            
            if (current_date - last_date).days > 5:
                logger.error("yfinance connectivity test failed: Data is not recent")
                return False
        
        return True
    except Exception as e:
        logger.error(f"yfinance connectivity test failed: {str(e)}")
        return False

# Add the verification check when the app starts
@app.before_first_request
def check_dependencies():
    """Check dependencies before first request"""
    if not verify_yfinance_connectivity():
        logger.error("yfinance connectivity check failed. Data fetching may not work properly.")
    else:
        logger.info("yfinance connectivity check passed successfully.")

if __name__ == '__main__':
    app.run(debug=True, port=5000) 