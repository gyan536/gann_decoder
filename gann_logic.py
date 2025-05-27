import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import math
import logging
import os
import requests
import json
from bsedata.bse import BSE
from rapidfuzz import process, fuzz
import random
from gann_analysis import GannAnalysis
import ta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StockSymbolDetector:
    def __init__(self):
        self.bse = None
        self.symbols_cache = {}
        self._initialize_bse()
        self._initialize_symbols()
    
    def _initialize_bse(self):
        """Initialize BSE connection and create necessary files"""
        try:
            # Create data directory if it doesn't exist
            data_dir = os.path.join(os.path.dirname(__file__), 'data')
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            
            # Create stk.json if it doesn't exist
            stk_path = os.path.join(data_dir, 'stk.json')
            if not os.path.exists(stk_path):
                # Create empty stock data file
                with open(stk_path, 'w') as f:
                    json.dump({}, f)
            
            self.bse = BSE(update_codes=True)
            logger.info("Successfully initialized BSE connection")
            
        except Exception as e:
            logger.error(f"Error initializing BSE: {str(e)}")
            raise
    
    def _initialize_symbols(self):
        """Initialize BSE symbols cache"""
        try:
            logger.info("Initializing BSE symbols cache...")
            if not self.bse:
                raise Exception("BSE connection not initialized")
                
            stocks = self.bse.getScripCodes()
            if stocks:
                self.symbols_cache = stocks
                logger.info(f"Loaded {len(stocks)} BSE symbols")
                
                # Save symbols to cache file
                cache_path = os.path.join(os.path.dirname(__file__), 'data', 'symbols_cache.json')
                with open(cache_path, 'w') as f:
                    json.dump(stocks, f)
            else:
                # Try loading from cache if available
                cache_path = os.path.join(os.path.dirname(__file__), 'data', 'symbols_cache.json')
                if os.path.exists(cache_path):
                    with open(cache_path, 'r') as f:
                        self.symbols_cache = json.load(f)
                    logger.info(f"Loaded {len(self.symbols_cache)} BSE symbols from cache")
                else:
                    logger.error("Failed to load BSE symbols and no cache available")
                    raise Exception("Failed to load BSE symbols")
                    
        except Exception as e:
            logger.error(f"Error initializing BSE symbols: {str(e)}")
            raise
    
    def detect_symbol(self, query):
        """Detect the correct stock symbol from a query string"""
        if not query:
            return None
            
        query = query.upper().strip()
        logger.info(f"Detecting symbol for query: {query}")
        
        # Direct symbol match
        for code, name in self.symbols_cache.items():
            if query == str(code) or query == name.upper():
                logger.info(f"Found direct match: {query} -> {code}")
                return code
        
        # Try fuzzy matching
        best_match = None
        best_score = 0
        
        for code, name in self.symbols_cache.items():
            score = fuzz.ratio(query, name.upper())
            if score > best_score and score > 75:
                best_score = score
                best_match = code
        
        if best_match:
            logger.info(f"Found fuzzy match: {query} -> {best_match} (score: {best_score})")
            return best_match
            
        logger.warning(f"No symbol match found for: {query}")
        return None

class MarketDataFetcher:
    def __init__(self):
        self.symbol_detector = StockSymbolDetector()
        self.bse = self.symbol_detector.bse
    
    def get_stock_data(self, query):
        """Fetch stock data from BSE"""
        try:
            # Detect symbol
            symbol = self.symbol_detector.detect_symbol(query)
            if not symbol:
                error_msg = f"Could not detect valid symbol for '{query}'. Please verify the company name or BSE code."
                logger.error(error_msg)
                return None, error_msg
            
            logger.info(f"Fetching data for BSE symbol: {symbol}")
            
            # Get quote and historical data
            quote = self.bse.getQuote(str(symbol))
            if not quote:
                error_msg = f"No data available for BSE symbol {symbol}"
                logger.error(error_msg)
                return None, error_msg
            
            # Convert quote to DataFrame format
            data = pd.DataFrame([{
                'Date': datetime.now().date(),
                'Open': float(quote.get('openPrice', 0)),
                'High': float(quote.get('dayHigh', 0)),
                'Low': float(quote.get('dayLow', 0)),
                'Close': float(quote.get('currentValue', 0)),
                'Volume': float(quote.get('totalTradedVolume', 0))
            }])
            
            # Get historical data for the last year
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            
            # TODO: Implement historical data fetching from BSE
            # For now, we'll use the current data point
            
            logger.info(f"Successfully fetched data for BSE symbol {symbol}")
            return data, None
            
        except Exception as e:
            error_msg = f"Error fetching data for {query}: {str(e)}"
            logger.error(error_msg)
            return None, error_msg

def calculate_gann_levels(price):
    """Calculate Gann Square of 9 price levels"""
    levels = []
    for i in range(-4, 5):
        level = price * (math.sqrt(9) ** i)
        levels.append(round(level, 2))
    return sorted(levels)

def calculate_technical_indicators(data):
    """Calculate technical indicators"""
    df = data.copy()
    
    # Moving Averages using ta
    df['MA20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['MA50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['MA200'] = ta.trend.sma_indicator(df['Close'], window=200)
    
    # RSI using ta
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    
    # MACD using ta
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['Signal_Line'] = macd.macd_signal()
    
    # Volume Analysis
    df['Volume_MA20'] = ta.trend.sma_indicator(df['Volume'], window=20)
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA20']
    
    return df

def analyze_market_structure(data):
    """Analyze market structure"""
    # Price Trends
    short_term_trend = 'BULLISH' if data['MA20'].iloc[-1] > data['MA50'].iloc[-1] else 'BEARISH'
    long_term_trend = 'BULLISH' if data['Close'].iloc[-1] > data['MA200'].iloc[-1] else 'BEARISH'
    
    # Support and Resistance
    recent_low = data['Low'].tail(20).min()
    recent_high = data['High'].tail(20).max()
    
    # Volume Analysis
    volume_trend = 'HIGH' if data['Volume_Ratio'].iloc[-1] > 1.5 else 'LOW' if data['Volume_Ratio'].iloc[-1] < 0.5 else 'NORMAL'
    
    # Momentum
    momentum = 'STRONG' if data['RSI'].iloc[-1] > 60 else 'WEAK' if data['RSI'].iloc[-1] < 40 else 'NEUTRAL'
    
    return {
        'short_term_trend': short_term_trend,
        'long_term_trend': long_term_trend,
        'support_level': round(recent_low, 2),
        'resistance_level': round(recent_high, 2),
        'volume_trend': volume_trend,
        'momentum': momentum
    }

def analyze_stock(stock_name):
    """
    Analyze stock using Gann Theory
    Returns comprehensive analysis with real data
    """
    try:
        # Initialize Gann analyzer
        analyzer = GannAnalysis()
        
        # Generate analysis
        analysis = analyzer.generate_gann_analysis(stock_name)
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing stock: {str(e)}")
        return {
            'stock': stock_name,
            'status': 'ERROR',
            'message': f'Analysis failed: {str(e)}'
        } 