import numpy as np
import pandas as pd
import math
from datetime import datetime, timedelta
import yfinance as yf
from nsepy import get_history
import ta
import json
import os
import requests
from bs4 import BeautifulSoup
import fuzz

class GannAnalysis:
    def __init__(self):
        self.fibonacci_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        self.gann_angles = [82.5, 75, 71.25, 63.75, 45, 26.25, 18.75, 15, 7.5]
        self.time_cycles = [90, 144, 180, 270, 360]
        self.symbols_cache = {}
        self._initialize_symbols()
    
    def _initialize_symbols(self):
        """Initialize and cache stock symbols"""
        cache_file = os.path.join(os.path.dirname(__file__), 'stock_symbols.json')
        
        # Try to load from cache first
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    self.symbols_cache = json.load(f)
                print("Loaded stock symbols from cache")
                return
            except:
                pass
        
        # If cache doesn't exist or is invalid, create new mapping
        self.symbols_cache = self._create_symbol_mapping()
        
        # Save to cache
        try:
            with open(cache_file, 'w') as f:
                json.dump(self.symbols_cache, f)
            print("Saved stock symbols to cache")
        except:
            print("Warning: Could not save symbols to cache")

    def _create_symbol_mapping(self):
        """Create comprehensive symbol mapping with enhanced NLP matching"""
        try:
            # Try to read the CSV file
            csv_path = os.path.join(os.path.dirname(__file__), 'indian_stocks.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                # Create mapping dictionary with enhanced variations
                mapping = {}
                for _, row in df.iterrows():
                    symbol = row['Symbol']
                    trading_symbol = row['Trading_Symbol']
                    company_name = row.get('Company_Name', '')
                    
                    # Store all variations
                    variations = {
                        symbol: trading_symbol,
                        symbol.upper(): trading_symbol,
                        symbol.lower(): trading_symbol
                    }
                    
                    if company_name:
                        # Add company name variations
                        variations[company_name] = trading_symbol
                        variations[company_name.upper()] = trading_symbol
                        variations[company_name.lower()] = trading_symbol
                        
                        # Add without spaces
                        variations[company_name.replace(' ', '')] = trading_symbol
                        
                        # Add without common suffixes
                        for suffix in [' LIMITED', ' LTD', ' INDIA', ' CORPORATION', ' CORP', ' PRIVATE', ' PVT']:
                            clean_name = company_name.replace(suffix, '').strip()
                            variations[clean_name] = trading_symbol
                            variations[clean_name.upper()] = trading_symbol
                            variations[clean_name.lower()] = trading_symbol
                        
                        # Add common abbreviations
                        words = company_name.split()
                        if len(words) > 1:
                            abbr = ''.join(word[0] for word in words if word[0].isalpha())
                            variations[abbr] = trading_symbol
                            variations[abbr.upper()] = trading_symbol
                    
                    mapping.update(variations)
                
                print(f"Loaded {len(mapping)} symbol variations from CSV")
                return mapping
            
            print("CSV file not found, using default mappings")
            return self._create_default_mapping()
        except Exception as e:
            print(f"Error loading CSV: {str(e)}, using default mappings")
            return self._create_default_mapping()
    
    def _create_default_mapping(self):
        """Create default symbol mapping for common stocks"""
        symbols = {}
        
        # Add Nifty 50 companies (most actively traded)
        nifty50_symbols = {
            'RELIANCE': ['RELIANCE INDUSTRIES', 'RIL'],
            'TCS': ['TATA CONSULTANCY SERVICES', 'TATA CONSULTANCY'],
            'HDFCBANK': ['HDFC BANK', 'HOUSING DEVELOPMENT FINANCE'],
            'INFY': ['INFOSYS', 'INFOSYS LIMITED'],
            'ICICIBANK': ['ICICI BANK'],
            'HINDUNILVR': ['HINDUSTAN UNILEVER', 'HUL'],
            'ITC': ['ITC LIMITED'],
            'SBIN': ['STATE BANK OF INDIA', 'SBI'],
            'BHARTIARTL': ['BHARTI AIRTEL', 'AIRTEL'],
            'KOTAKBANK': ['KOTAK MAHINDRA BANK', 'KOTAK BANK'],
            'LT': ['LARSEN & TOUBRO', 'L&T', 'LARSEN AND TOUBRO'],
            'HCLTECH': ['HCL TECHNOLOGIES', 'HCL TECH'],
            'ASIANPAINT': ['ASIAN PAINTS'],
            'AXISBANK': ['AXIS BANK'],
            'MARUTI': ['MARUTI SUZUKI', 'MARUTI SUZUKI INDIA'],
            'BAJFINANCE': ['BAJAJ FINANCE'],
            'WIPRO': ['WIPRO LIMITED'],
            'TATASTEEL': ['TATA STEEL'],
            'TATAMOTORS': ['TATA MOTORS'],
            'TECHM': ['TECH MAHINDRA'],
            'ULTRACEMCO': ['ULTRATECH CEMENT'],
            'TITAN': ['TITAN COMPANY'],
            'NESTLEIND': ['NESTLE INDIA'],
            'POWERGRID': ['POWER GRID CORPORATION', 'POWER GRID'],
            'NTPC': ['NTPC LIMITED'],
            'M&M': ['MAHINDRA & MAHINDRA', 'MAHINDRA AND MAHINDRA'],
            'BAJAJ-AUTO': ['BAJAJ AUTO'],
            'SUNPHARMA': ['SUN PHARMACEUTICAL', 'SUN PHARMA'],
            'ONGC': ['OIL AND NATURAL GAS CORPORATION'],
            'COALINDIA': ['COAL INDIA'],
            'SHREECEM': ['SHREE CEMENT', 'SHREE CEMENTS', 'SHREE CEMENT LIMITED', 'SHREE CEMENTS LTD', 'SHREE CEMENTS LIMITED']
        }
        symbols.update(nifty50_symbols)
        
        # Add other major companies
        other_major_symbols = {
            'ZOMATO': ['ZOMATO LIMITED'],
            'PAYTM': ['PAYTM', 'ONE97 COMMUNICATIONS'],
            'NYKAA': ['FSN E-COMMERCE', 'NYKAA'],
            'POLICYBZR': ['PB FINTECH', 'POLICY BAZAAR'],
            'DMART': ['AVENUE SUPERMARTS', 'D-MART'],
            'IRCTC': ['INDIAN RAILWAY CATERING', 'IRCTC'],
            'BANDHANBNK': ['BANDHAN BANK'],
            'PNB': ['PUNJAB NATIONAL BANK'],
            'YESBANK': ['YES BANK'],
            'IDEA': ['VODAFONE IDEA'],
            'SUZLON': ['SUZLON ENERGY'],
            'TATAPOWER': ['TATA POWER'],
            'ADANIENT': ['ADANI ENTERPRISES'],
            'ADANIPORTS': ['ADANI PORTS'],
            'ADANIGREEN': ['ADANI GREEN ENERGY'],
            'ADANIPOWER': ['ADANI POWER'],
        }
        symbols.update(other_major_symbols)
        
        # Create reverse mappings
        reverse_symbols = {}
        for symbol, names in symbols.items():
            for name in names:
                reverse_symbols[name.upper()] = symbol
                # Add variations with common suffixes
                for suffix in [' LTD', ' LIMITED', ' INDIA', ' CORPORATION']:
                    reverse_symbols[name.upper() + suffix] = symbol
        
        # Merge both mappings
        final_mapping = {**symbols, **reverse_symbols}
        return final_mapping

    def _format_symbol(self, input_symbol):
        """Format and clean the input symbol using enhanced NLP matching"""
        # Clean and normalize input
        clean_input = input_symbol.upper().strip()
        clean_input = clean_input.replace('.NS', '').replace('.BO', '')
        clean_input = clean_input.replace('NSE:', '').replace('BSE:', '')
        
        # Try direct match first
        if clean_input in self.symbols_cache:
            return self.symbols_cache[clean_input]
        
        # Remove common suffixes and try again
        common_suffixes = [
            ' LIMITED', ' LTD', ' LTD.', ' LIMITED.', ' INDIA', 
            ' INDIA LIMITED', ' CORPORATION', ' CORP', ' PRIVATE',
            ' PVT', ' PVT.', ' INDUSTRIES', ' PRODUCTS'
        ]
        
        test_symbol = clean_input
        for suffix in common_suffixes:
            if test_symbol.endswith(suffix):
                test_symbol = test_symbol[:-len(suffix)].strip()
                if test_symbol in self.symbols_cache:
                    return self.symbols_cache[test_symbol]
        
        # Try fuzzy matching using token sort ratio
        best_match = None
        best_score = 0
        input_tokens = set(clean_input.split())
        
        for company, symbol in self.symbols_cache.items():
            # Skip numeric-only keys
            if company.isdigit():
                continue
                
            company_tokens = set(company.split())
            
            # Calculate token overlap score
            common_tokens = input_tokens & company_tokens
            if common_tokens:
                # Use fuzz.token_sort_ratio for better matching
                score = fuzz.token_sort_ratio(clean_input, company)
                
                # Boost score if common tokens found
                score += len(common_tokens) * 10
                
                # Boost score if input is subset of company name
                if input_tokens.issubset(company_tokens):
                    score += 20
                
                if score > best_score:
                    best_score = score
                    best_match = symbol
        
        # If good match found
        if best_score >= 60:
            print(f"Found fuzzy match for '{input_symbol}' with score {best_score}")
            return best_match
        
        # If no match found, try to extract potential symbol
        # Look for capital letter sequences
        potential_symbol = ''.join(c for c in clean_input if c.isupper())
        if len(potential_symbol) >= 2 and potential_symbol in self.symbols_cache:
            return self.symbols_cache[potential_symbol]
        
        # If still no match, clean up and return the original
        return ''.join(e for e in clean_input if e.isalnum())

    def get_stock_data(self, symbol, period='1y'):
        """Fetch stock data from multiple sources with improved error handling"""
        # Format the symbol
        clean_symbol = self._format_symbol(symbol)
        errors = []
        exchange_data = {'NSE': None, 'BSE': None}
        
        print(f"Processing request for: {symbol}")
        print(f"Mapped to symbol: {clean_symbol}")
        
        # Try NSE first
        try:
            print(f"Attempting NSE data fetch for: {clean_symbol}")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            
            try:
                nse_data = get_history(symbol=clean_symbol, start=start_date, end=end_date)
                if not nse_data.empty and len(nse_data) > 0:
                    print(f"Successfully fetched NSE data for: {clean_symbol}")
                    exchange_data['NSE'] = nse_data
            except Exception as nse_error:
                errors.append(f"NSE: {str(nse_error)}")
                print(f"NSE fetch failed, trying BSE...")
        except Exception as e:
            errors.append(f"NSE setup: {str(e)}")
        
        # Try BSE through Yahoo Finance
        try:
            bse_symbol = clean_symbol + '.BO'
            print(f"Attempting BSE data fetch for: {bse_symbol}")
            stock = yf.Ticker(bse_symbol)
            bse_data = stock.history(period=period)
            if not bse_data.empty and len(bse_data) > 0:
                print(f"Successfully fetched BSE data for: {bse_symbol}")
                exchange_data['BSE'] = bse_data
        except Exception as e:
            errors.append(f"BSE: {str(e)}")
            print(f"BSE fetch failed...")

        # Try NSE through Yahoo Finance if direct NSE fetch failed
        if exchange_data['NSE'] is None:
            try:
                nse_symbol = clean_symbol + '.NS'
                print(f"Attempting Yahoo Finance NSE data fetch for: {nse_symbol}")
                stock = yf.Ticker(nse_symbol)
                nse_data = stock.history(period=period)
                if not nse_data.empty and len(nse_data) > 0:
                    print(f"Successfully fetched NSE data from Yahoo for: {nse_symbol}")
                    exchange_data['NSE'] = nse_data
            except Exception as e:
                errors.append(f"Yahoo NSE: {str(e)}")
                print(f"Yahoo NSE fetch failed...")

        # Compare and analyze exchange data
        if exchange_data['NSE'] is not None and exchange_data['BSE'] is not None:
            # Calculate price difference percentage
            nse_latest = exchange_data['NSE']['Close'].iloc[-1]
            bse_latest = exchange_data['BSE']['Close'].iloc[-1]
            price_diff_pct = abs(nse_latest - bse_latest) / nse_latest * 100
            
            # Compare volumes
            nse_volume = exchange_data['NSE']['Volume'].mean()
            bse_volume = exchange_data['BSE']['Volume'].mean()
            
            print(f"\nExchange Comparison for {clean_symbol}:")
            print(f"NSE Price: {nse_latest:.2f} | BSE Price: {bse_latest:.2f}")
            print(f"Price Difference: {price_diff_pct:.2f}%")
            print(f"NSE Avg Volume: {nse_volume:.0f} | BSE Avg Volume: {bse_volume:.0f}")
            
            # Use NSE data if available (typically more liquid)
            if exchange_data['NSE'] is not None:
                print("Using NSE data for analysis (higher liquidity)")
                return exchange_data['NSE']
            return exchange_data['BSE']
        
        # If only one exchange has data, use that
        if exchange_data['NSE'] is not None:
            print("Only NSE data available, using NSE")
            return exchange_data['NSE']
        if exchange_data['BSE'] is not None:
            print("Only BSE data available, using BSE")
            return exchange_data['BSE']
        
        # If all attempts fail, raise an error with details
        error_msg = (f"Could not fetch data for '{symbol}' (mapped to '{clean_symbol}'). "
                    f"Please verify the company name or stock symbol. Errors encountered:\n")
        error_msg += "\n".join(errors)
        raise Exception(error_msg)

    def calculate_square_of_9(self, price):
        """Calculate Gann Square of 9 price levels"""
        levels = []
        sqrt9 = math.sqrt(9)
        
        # Calculate 8 price levels (4 above, 4 below)
        for i in range(-4, 5):
            level = price * (sqrt9 ** i)
            levels.append(round(level, 2))
        
        return sorted(levels)

    def calculate_gann_angles(self, data):
        """Calculate Gann angle lines"""
        high = data['High'].max()
        low = data['Low'].min()
        price_range = high - low
        time_range = len(data)
        
        angles = {}
        for angle in self.gann_angles:
            # Convert angle to radians
            rad = math.radians(angle)
            # Calculate slope
            slope = math.tan(rad)
            # Calculate price movement per time unit
            price_move = slope * price_range / time_range
            angles[angle] = price_move
            
        return angles

    def find_time_cycles(self, data):
        """Identify time cycle patterns"""
        cycles = {}
        close_prices = data['Close']
        total_length = len(close_prices)
        
        for cycle in self.time_cycles:
            try:
                if total_length >= 2 * cycle:  # Need at least 2 cycles worth of data
                    # Get the most recent cycle and the one before it
                    current_cycle = close_prices[-cycle:].values
                    previous_cycle = close_prices[-2*cycle:-cycle].values
                    
                    # Ensure both arrays have the same length
                    min_length = min(len(current_cycle), len(previous_cycle))
                    if min_length > 0:
                        current_cycle = current_cycle[-min_length:]
                        previous_cycle = previous_cycle[-min_length:]
                        
                        # Calculate correlation
                        correlation = np.corrcoef(current_cycle, previous_cycle)[0,1]
                        if not np.isnan(correlation):
                            cycles[cycle] = round(correlation, 2)
                        else:
                            cycles[cycle] = 0.0
                    else:
                        cycles[cycle] = 0.0
                else:
                    cycles[cycle] = 0.0
            except Exception as e:
                print(f"Error calculating {cycle}-day cycle: {str(e)}")
                cycles[cycle] = 0.0
        
        return cycles

    def calculate_support_resistance(self, data, price):
        """Calculate support and resistance levels using Gann methods"""
        levels = self.calculate_square_of_9(price)
        recent_high = data['High'].tail(20).max()
        recent_low = data['Low'].tail(20).min()
        
        support_levels = [level for level in levels if level < price]
        resistance_levels = [level for level in levels if level > price]
        
        return {
            'current_price': price,
            'support_levels': support_levels,
            'resistance_levels': resistance_levels,
            'recent_high': recent_high,
            'recent_low': recent_low
        }

    def analyze_price_patterns(self, data):
        """Analyze price patterns using Gann principles"""
        df = data.copy()
        
        # Calculate technical indicators using ta library
        # SMA
        df['SMA20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['SMA50'] = ta.trend.sma_indicator(df['Close'], window=50)
        
        # RSI
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        
        # Additional indicators
        df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=14)
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        
        # Get current values
        current_price = df['Close'].iloc[-1]
        current_sma20 = df['SMA20'].iloc[-1]
        current_sma50 = df['SMA50'].iloc[-1]
        current_rsi = df['RSI'].iloc[-1]
        current_adx = df['ADX'].iloc[-1]
        current_macd = df['MACD'].iloc[-1]
        
        # Determine trend with multiple confirmations
        trend = 'BULLISH' if (current_sma20 > current_sma50 and current_macd > 0) else 'BEARISH'
        
        # Check momentum with RSI and ADX
        momentum = 'OVERBOUGHT' if current_rsi > 70 else 'OVERSOLD' if current_rsi < 30 else 'NEUTRAL'
        trend_strength = 'STRONG' if current_adx > 25 else 'WEAK'
        
        return {
            'trend': trend,
            'momentum': momentum,
            'trend_strength': trend_strength,
            'rsi': round(current_rsi, 2),
            'adx': round(current_adx, 2),
            'sma20': round(current_sma20, 2),
            'sma50': round(current_sma50, 2)
        }

    def analyze_exceeding_moves(self, data):
        """Analyze when price swings exceed previous moves in length or duration"""
        swings = []
        current_swing = {'start_idx': 0, 'direction': None}
        
        # Detect price swings
        for i in range(1, len(data)):
            if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
                if current_swing['direction'] == 'down':
                    swings.append({
                        'start_idx': current_swing['start_idx'],
                        'end_idx': i-1,
                        'direction': 'down',
                        'price_range': abs(data['Close'].iloc[current_swing['start_idx']] - data['Close'].iloc[i-1]),
                        'duration': i - current_swing['start_idx']
                    })
                    current_swing = {'start_idx': i-1, 'direction': 'up'}
                elif current_swing['direction'] is None:
                    current_swing['direction'] = 'up'
            elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
                if current_swing['direction'] == 'up':
                    swings.append({
                        'start_idx': current_swing['start_idx'],
                        'end_idx': i-1,
                        'direction': 'up',
                        'price_range': abs(data['Close'].iloc[current_swing['start_idx']] - data['Close'].iloc[i-1]),
                        'duration': i - current_swing['start_idx']
                    })
                    current_swing = {'start_idx': i-1, 'direction': 'down'}
                elif current_swing['direction'] is None:
                    current_swing['direction'] = 'down'
        
        # Analyze exceeding moves
        signals = []
        for i in range(1, len(swings)):
            prev_swing = swings[i-1]
            curr_swing = swings[i]
            
            # Compare price ranges
            if curr_swing['price_range'] > prev_swing['price_range'] * 1.2:  # 20% larger
                signals.append({
                    'type': 'EXCEEDING_MOVE_PRICE',
                    'direction': curr_swing['direction'],
                    'magnitude': round(curr_swing['price_range'] / prev_swing['price_range'], 2),
                    'date': data.index[curr_swing['end_idx']]
                })
            
            # Compare durations
            if curr_swing['duration'] > prev_swing['duration'] * 1.5:  # 50% longer
                signals.append({
                    'type': 'EXCEEDING_MOVE_TIME',
                    'direction': curr_swing['direction'],
                    'magnitude': round(curr_swing['duration'] / prev_swing['duration'], 2),
                    'date': data.index[curr_swing['end_idx']]
                })
        
        return signals

    def analyze_trading_ranges(self, data, window=20):
        """Detect trading ranges and potential breakouts"""
        ranges = []
        current_range = None
        std_threshold = 0.015  # 1.5% standard deviation threshold
        
        for i in range(window, len(data)):
            window_data = data.iloc[i-window:i]
            price_std = window_data['Close'].std() / window_data['Close'].mean()
            
            if price_std < std_threshold:
                if current_range is None:
                    current_range = {
                        'start_idx': i-window,
                        'support': window_data['Low'].min(),
                        'resistance': window_data['High'].max(),
                        'avg_volume': window_data['Volume'].mean()
                    }
            else:
                if current_range is not None:
                    current_range['end_idx'] = i
                    ranges.append(current_range)
                    current_range = None
        
        # Detect breakouts
        breakouts = []
        for r in ranges:
            range_end_idx = r['end_idx']
            if range_end_idx + 1 < len(data):
                close = data['Close'].iloc[range_end_idx + 1]
                volume = data['Volume'].iloc[range_end_idx + 1]
                
                if close > r['resistance'] and volume > r['avg_volume'] * 1.5:
                    breakouts.append({
                        'type': 'RANGE_BREAKOUT_UP',
                        'date': data.index[range_end_idx + 1],
                        'range_size': round(r['resistance'] - r['support'], 2),
                        'volume_ratio': round(volume / r['avg_volume'], 2)
                    })
                elif close < r['support'] and volume > r['avg_volume'] * 1.5:
                    breakouts.append({
                        'type': 'RANGE_BREAKOUT_DOWN',
                        'date': data.index[range_end_idx + 1],
                        'range_size': round(r['resistance'] - r['support'], 2),
                        'volume_ratio': round(volume / r['avg_volume'], 2)
                    })
        
        return breakouts

    def analyze_old_levels(self, data, lookback_days=365):
        """Analyze old support/resistance levels"""
        # Find significant swing points
        highs = []
        lows = []
        window = 10
        
        for i in range(window, len(data)-window):
            if all(data['High'].iloc[i] > data['High'].iloc[i-window:i]) and \
               all(data['High'].iloc[i] > data['High'].iloc[i+1:i+window+1]):
                highs.append({
                    'price': data['High'].iloc[i],
                    'date': data.index[i]
                })
            
            if all(data['Low'].iloc[i] < data['Low'].iloc[i-window:i]) and \
               all(data['Low'].iloc[i] < data['Low'].iloc[i+1:i+window+1]):
                lows.append({
                    'price': data['Low'].iloc[i],
                    'date': data.index[i]
                })
        
        # Find current approaches to old levels
        signals = []
        current_price = data['Close'].iloc[-1]
        threshold = 0.02  # 2% proximity threshold
        
        for high in highs:
            if abs(current_price - high['price']) / high['price'] < threshold:
                signals.append({
                    'type': 'APPROACHING_OLD_TOP',
                    'level': round(high['price'], 2),
                    'original_date': high['date'],
                    'proximity_pct': round(abs(current_price - high['price']) / high['price'] * 100, 2)
                })
        
        for low in lows:
            if abs(current_price - low['price']) / low['price'] < threshold:
                signals.append({
                    'type': 'APPROACHING_OLD_BOTTOM',
                    'level': round(low['price'], 2),
                    'original_date': low['date'],
                    'proximity_pct': round(abs(current_price - low['price']) / low['price'] * 100, 2)
                })
        
        return signals

    def analyze_rapid_moves(self, data, window=20):
        """Analyze rapid price moves and emotional spikes"""
        signals = []
        
        # Calculate average candle size and volume
        data['body_size'] = abs(data['Close'] - data['Open'])
        avg_body_size = data['body_size'].rolling(window=window).mean()
        avg_volume = data['Volume'].rolling(window=window).mean()
        
        for i in range(window, len(data)):
            current_body = data['body_size'].iloc[i]
            current_volume = data['Volume'].iloc[i]
            
            # Check for price spikes
            if current_body > avg_body_size.iloc[i] * 2.5:  # 250% larger than average
                rsi = ta.momentum.rsi(data['Close'], window=14).iloc[i]
                signals.append({
                    'type': 'EMOTIONAL_SPIKE',
                    'date': data.index[i],
                    'magnitude': round(current_body / avg_body_size.iloc[i], 2),
                    'direction': 'up' if data['Close'].iloc[i] > data['Open'].iloc[i] else 'down',
                    'volume_surge': round(current_volume / avg_volume.iloc[i], 2),
                    'rsi': round(rsi, 2)
                })
        
        return signals

    def analyze_closing_prices(self, data):
        """Analyze significance of closing prices"""
        signals = []
        window = 20
        
        # Calculate key levels
        data['SMA20'] = ta.trend.sma_indicator(data['Close'], window=20)
        data['Upper_Band'] = data['SMA20'] + data['Close'].rolling(window=window).std() * 2
        data['Lower_Band'] = data['SMA20'] - data['Close'].rolling(window=window).std() * 2
        
        for i in range(window, len(data)):
            close = data['Close'].iloc[i]
            open_price = data['Open'].iloc[i]
            high = data['High'].iloc[i]
            low = data['Low'].iloc[i]
            
            # Strong close analysis
            if close > data['Upper_Band'].iloc[i]:
                signals.append({
                    'type': 'STRONG_CLOSE_ABOVE_BANDS',
                    'date': data.index[i],
                    'close_strength': round((close - data['SMA20'].iloc[i]) / data['SMA20'].iloc[i] * 100, 2)
                })
            elif close < data['Lower_Band'].iloc[i]:
                signals.append({
                    'type': 'STRONG_CLOSE_BELOW_BANDS',
                    'date': data.index[i],
                    'close_strength': round((data['SMA20'].iloc[i] - close) / data['SMA20'].iloc[i] * 100, 2)
                })
            
            # Close position in candle
            total_range = high - low
            if total_range > 0:
                close_position = (close - low) / total_range
                if close_position > 0.9:  # Close near high
                    signals.append({
                        'type': 'CLOSE_NEAR_HIGH',
                        'date': data.index[i],
                        'strength': round(close_position * 100, 2)
                    })
                elif close_position < 0.1:  # Close near low
                    signals.append({
                        'type': 'CLOSE_NEAR_LOW',
                        'date': data.index[i],
                        'strength': round((1 - close_position) * 100, 2)
                    })
        
        return signals

    def analyze_gann_numbers(self, data):
        """Analyze price and time relationships using Gann's favorite numbers"""
        gann_numbers = [3, 7, 9, 12, 30, 45, 60, 90, 120, 180, 360]
        signals = []
        
        # Get swing points
        highs = []
        lows = []
        window = 10
        
        for i in range(window, len(data)-window):
            if all(data['High'].iloc[i] > data['High'].iloc[i-window:i]) and \
               all(data['High'].iloc[i] > data['High'].iloc[i+1:i+window+1]):
                highs.append({
                    'index': i,
                    'price': data['High'].iloc[i],
                    'date': data.index[i]
                })
            
            if all(data['Low'].iloc[i] < data['Low'].iloc[i-window:i]) and \
               all(data['Low'].iloc[i] < data['Low'].iloc[i+1:i+window+1]):
                lows.append({
                    'index': i,
                    'price': data['Low'].iloc[i],
                    'date': data.index[i]
                })
        
        # Analyze time cycles
        current_idx = len(data) - 1
        for point in highs + lows:
            bars_since = current_idx - point['index']
            
            # Check if current bar count matches any Gann number
            for gann_num in gann_numbers:
                if abs(bars_since - gann_num) <= 2:  # Allow 2-bar margin
                    signals.append({
                        'type': 'GANN_TIME_CYCLE',
                        'bars': bars_since,
                        'gann_number': gann_num,
                        'reference_date': point['date'],
                        'reference_price': point['price']
                    })
        
        # Analyze price relationships
        current_price = data['Close'].iloc[-1]
        for point in highs + lows:
            price_diff = abs(current_price - point['price'])
            price_ratio = price_diff / point['price'] * 100
            
            # Check if price difference matches Gann numbers percentage
            for gann_num in gann_numbers:
                if abs(price_ratio - gann_num) <= 1:  # 1% margin
                    signals.append({
                        'type': 'GANN_PRICE_LEVEL',
                        'percentage': price_ratio,
                        'gann_number': gann_num,
                        'reference_price': point['price']
                    })
        
        return signals

    def analyze_tops_and_bottoms(self, data, proximity_pct=2):
        """Detect double and triple tops/bottoms"""
        signals = []
        window = 20
        proximity_threshold = proximity_pct / 100
        
        # Find significant swing points
        swing_points = []
        for i in range(window, len(data)-window):
            if all(data['High'].iloc[i] > data['High'].iloc[i-window:i]) and \
               all(data['High'].iloc[i] > data['High'].iloc[i+1:i+window+1]):
                swing_points.append({
                    'type': 'HIGH',
                    'index': i,
                    'price': data['High'].iloc[i],
                    'date': data.index[i]
                })
            elif all(data['Low'].iloc[i] < data['Low'].iloc[i-window:i]) and \
                 all(data['Low'].iloc[i] < data['Low'].iloc[i+1:i+window+1]):
                swing_points.append({
                    'type': 'LOW',
                    'index': i,
                    'price': data['Low'].iloc[i],
                    'date': data.index[i]
                })
        
        # Analyze swing points for patterns
        for i in range(len(swing_points)):
            current = swing_points[i]
            matches = []
            
            # Look for similar levels
            for j in range(i+1, len(swing_points)):
                other = swing_points[j]
                if current['type'] == other['type']:
                    price_diff = abs(current['price'] - other['price']) / current['price']
                    if price_diff <= proximity_threshold:
                        matches.append(other)
            
            # Check for patterns
            if len(matches) == 1:
                pattern_type = f"DOUBLE_{current['type']}"
                confidence = (1 - price_diff) * 100
                signals.append({
                    'type': pattern_type,
                    'confidence': round(confidence, 2),
                    'first_date': current['date'],
                    'second_date': matches[0]['date'],
                    'price_level': round((current['price'] + matches[0]['price']) / 2, 2)
                })
            elif len(matches) == 2:
                pattern_type = f"TRIPLE_{current['type']}"
                avg_price = (current['price'] + matches[0]['price'] + matches[1]['price']) / 3
                confidence = (1 - max(abs(p['price'] - avg_price) for p in [current] + matches) / avg_price) * 100
                signals.append({
                    'type': pattern_type,
                    'confidence': round(confidence, 2),
                    'first_date': current['date'],
                    'second_date': matches[0]['date'],
                    'third_date': matches[1]['date'],
                    'price_level': round(avg_price, 2)
                })
        
        return signals

    def analyze_historical_zones(self, data, lookback_days=365):
        """Analyze historical price zones using 'look to the left' principle"""
        signals = []
        
        # Calculate price distribution
        price_range = data['High'].max() - data['Low'].min()
        num_bins = 50
        hist, bins = np.histogram(data['Close'], bins=num_bins)
        
        # Find zones with high activity
        active_zones = []
        for i in range(len(hist)):
            if hist[i] > np.mean(hist) + np.std(hist):
                zone = {
                    'min_price': bins[i],
                    'max_price': bins[i+1],
                    'touches': hist[i],
                    'reversals': 0
                }
                
                # Count reversals in this zone
                for j in range(1, len(data)-1):
                    price = data['Close'].iloc[j]
                    if zone['min_price'] <= price <= zone['max_price']:
                        prev_close = data['Close'].iloc[j-1]
                        next_close = data['Close'].iloc[j+1]
                        if (prev_close < price and next_close < price) or \
                           (prev_close > price and next_close > price):
                            zone['reversals'] += 1
                
                active_zones.append(zone)
        
        # Generate signals for significant zones
        current_price = data['Close'].iloc[-1]
        for zone in active_zones:
            if zone['reversals'] >= 3:  # Minimum 3 reversals to be significant
                proximity = min(
                    abs(current_price - zone['min_price']),
                    abs(current_price - zone['max_price'])
                ) / price_range * 100
                
                signals.append({
                    'type': 'HISTORICAL_ZONE',
                    'min_price': round(zone['min_price'], 2),
                    'max_price': round(zone['max_price'], 2),
                    'touches': int(zone['touches']),
                    'reversals': zone['reversals'],
                    'proximity_pct': round(proximity, 2)
                })
        
        return signals

    def analyze_gann_patterns(self, data):
        """Analyze advanced Gann patterns in price data"""
        patterns = []
        
        # Get price data
        closes = data['Close'].values
        highs = data['High'].values
        lows = data['Low'].values
        dates = data.index
        
        # Scale factor for price comparisons (0.5% tolerance)
        price_scale = np.mean(closes) * 0.005
        
        # 1. Natural Square Pattern (1, 4, 9, 16, 25, 36, 49, 64, 81)
        natural_squares = [i * i for i in range(1, 10)]
        scaled_squares = [square * price_scale for square in natural_squares]
        
        for i in range(len(closes) - 1):
            for square in scaled_squares:
                if abs(closes[i] - square) < price_scale:
                    patterns.append({
                        'name': f'Natural Square {square/price_scale:.0f}',
                        'price': closes[i],
                        'date': dates[i].strftime('%Y-%m-%d'),
                        'strength': 'Strong',
                        'type': 'Square'
                    })

        # 2. Gann's Key Price Levels
        key_numbers = [0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618]
        price_range = max(highs) - min(lows)
        base_price = min(lows)
        
        for i in range(len(closes)):
            for ratio in key_numbers:
                level = base_price + (price_range * ratio)
                if abs(closes[i] - level) < price_scale:
                    patterns.append({
                        'name': f'Key Level {ratio:.3f}',
                        'price': closes[i],
                        'date': dates[i].strftime('%Y-%m-%d'),
                        'strength': 'Very Strong',
                        'type': 'Key Level'
                    })

        # 3. Price Range Divisions
        divisions = [0.25, 0.333, 0.5, 0.667, 0.75]
        for div in divisions:
            level = base_price + (price_range * div)
            for i in range(len(closes)):
                if abs(closes[i] - level) < price_scale:
                    patterns.append({
                        'name': f'Range Division {int(div * 100)}%',
                        'price': closes[i],
                        'date': dates[i].strftime('%Y-%m-%d'),
                        'strength': 'Moderate',
                        'type': 'Range'
                    })

        # 4. Gann Angles
        start_price = closes[0]
        for i in range(1, len(closes)):
            days = i
            
            # 1x1 angle (45 degrees)
            angle_1x1 = start_price + (days * price_scale)
            if abs(closes[i] - angle_1x1) < price_scale:
                patterns.append({
                    'name': '1x1 Gann Angle',
                    'price': closes[i],
                    'date': dates[i].strftime('%Y-%m-%d'),
                    'strength': 'Very Strong',
                    'type': 'Angle'
                })
            
            # 2x1 angle (63.75 degrees)
            angle_2x1 = start_price + (days * price_scale * 2)
            if abs(closes[i] - angle_2x1) < price_scale:
                patterns.append({
                    'name': '2x1 Gann Angle',
                    'price': closes[i],
                    'date': dates[i].strftime('%Y-%m-%d'),
                    'strength': 'Strong',
                    'type': 'Angle'
                })
            
            # 1x2 angle (26.25 degrees)
            angle_1x2 = start_price + (days * price_scale * 0.5)
            if abs(closes[i] - angle_1x2) < price_scale:
                patterns.append({
                    'name': '1x2 Gann Angle',
                    'price': closes[i],
                    'date': dates[i].strftime('%Y-%m-%d'),
                    'strength': 'Strong',
                    'type': 'Angle'
                })

        # 5. Time Cycle Patterns
        cycle_days = [90, 120, 180, 270, 360]
        for days in cycle_days:
            if len(closes) >= days:
                for i in range(0, len(closes) - days, days):
                    cycle_data = closes[i:i + days]
                    cycle_high = max(cycle_data)
                    cycle_low = min(cycle_data)
                    
                    # Check for repeating highs
                    if i + days < len(closes):
                        next_cycle = closes[i + days:min(i + 2*days, len(closes))]
                        next_high = max(next_cycle)
                        if abs(cycle_high - next_high) < price_scale:
                            patterns.append({
                                'name': f'{days}-Day Cycle High',
                                'price': next_high,
                                'date': dates[i + days + list(next_cycle).index(next_high)].strftime('%Y-%m-%d'),
                                'strength': 'Strong',
                                'type': 'Cycle'
                            })

        # 6. Square of Nine Analysis
        current_price = closes[-1]
        sq9_levels = self.calculate_square_of_9(current_price)
        for level in sq9_levels:
            if abs(current_price - level) < price_scale:
                patterns.append({
                    'name': 'Square of Nine Level',
                    'price': level,
                    'date': dates[-1].strftime('%Y-%m-%d'),
                    'strength': 'Very Strong',
                    'type': 'Square of Nine'
                })

        # 7. Price Clusters
        if len(patterns) > 0:
            # Group patterns by price level
            price_groups = {}
            for pattern in patterns:
                price_key = round(pattern['price'] / price_scale) * price_scale
                if price_key not in price_groups:
                    price_groups[price_key] = []
                price_groups[price_key].append(pattern)
            
            # Add cluster patterns
            for price_key, group in price_groups.items():
                if len(group) >= 3:
                    patterns.append({
                        'name': f'Pattern Cluster ({len(group)} patterns)',
                        'price': price_key,
                        'date': dates[-1].strftime('%Y-%m-%d'),
                        'strength': 'Very Strong',
                        'type': 'Cluster'
                    })

        # Sort patterns by date
        patterns.sort(key=lambda x: datetime.strptime(x['date'], '%Y-%m-%d'))
        
        # Add pattern count and summary
        if patterns:
            summary = {
                'total_patterns': len(patterns),
                'pattern_types': {
                    'Square': len([p for p in patterns if p['type'] == 'Square']),
                    'Key Level': len([p for p in patterns if p['type'] == 'Key Level']),
                    'Range': len([p for p in patterns if p['type'] == 'Range']),
                    'Angle': len([p for p in patterns if p['type'] == 'Angle']),
                    'Cycle': len([p for p in patterns if p['type'] == 'Cycle']),
                    'Square of Nine': len([p for p in patterns if p['type'] == 'Square of Nine']),
                    'Cluster': len([p for p in patterns if p['type'] == 'Cluster'])
                },
                'strength_distribution': {
                    'Very Strong': len([p for p in patterns if p['strength'] == 'Very Strong']),
                    'Strong': len([p for p in patterns if p['strength'] == 'Strong']),
                    'Moderate': len([p for p in patterns if p['strength'] == 'Moderate'])
                }
            }
            patterns.append(summary)
        
        return patterns

    def analyze_time_price_balance(self, data):
        """Analyze time-price relationships using Gann's methods"""
        balance_points = []
        
        # Get closing prices and dates
        closes = data['Close'].values
        dates = data.index
        highs = data['High'].values
        lows = data['Low'].values
        
        # 1. Price-Time Squares
        for i in range(len(closes) - 1):
            days_from_start = (dates[i] - dates[0]).days
            if abs(closes[i] - days_from_start) < 1:
                balance_points.append({
                    'type': 'Price-Time Square',
                    'price': closes[i],
                    'date': dates[i].strftime('%Y-%m-%d'),
                    'significance': 'High'
                })

        # 2. Gann's 1x1 Balance Line
        start_price = closes[0]
        for i in range(1, len(closes)):
            days_passed = i
            expected_1x1 = start_price + days_passed
            if abs(closes[i] - expected_1x1) < 0.5:
                balance_points.append({
                    'type': '1x1 Balance',
                    'price': closes[i],
                    'date': dates[i].strftime('%Y-%m-%d'),
                    'significance': 'Very High'
                })

        # 3. Time Symmetry Points
        for i in range(len(closes) - 1):
            for j in range(i + 1, len(closes)):
                time_diff = (dates[j] - dates[i]).days
                if time_diff in [45, 90, 180, 270, 360]:
                    if abs(closes[j] - closes[i]) < (max(closes) - min(closes)) * 0.02:  # 2% tolerance
                        balance_points.append({
                            'type': f'{time_diff}-Day Symmetry',
                            'price': closes[j],
                            'date': dates[j].strftime('%Y-%m-%d'),
                            'significance': 'Moderate'
                        })

        # 4. Price Range Balance Points
        total_range = max(highs) - min(lows)
        range_divisions = [0.382, 0.5, 0.618, 0.786]  # Fibonacci ratios
        base = min(lows)
        
        for div in range_divisions:
            balance_level = base + (total_range * div)
            for i in range(len(closes)):
                if abs(closes[i] - balance_level) < (total_range * 0.01):
                    balance_points.append({
                        'type': f'Range Balance {int(div * 100)}%',
                        'price': closes[i],
                        'date': dates[i].strftime('%Y-%m-%d'),
                        'significance': 'High'
                    })

        # 5. Time-Price Trend Balance
        if len(closes) >= 90:  # At least 90 days of data
            for window in [90, 180, 360]:
                if len(closes) >= window:
                    for i in range(len(closes) - window):
                        period_data = closes[i:i + window]
                        period_range = max(period_data) - min(period_data)
                        mid_point = min(period_data) + (period_range / 2)
                        
                        if abs(closes[i + window - 1] - mid_point) < (period_range * 0.02):
                            balance_points.append({
                                'type': f'{window}-Day Trend Balance',
                                'price': closes[i + window - 1],
                                'date': dates[i + window - 1].strftime('%Y-%m-%d'),
                                'significance': 'High'
                            })

        return balance_points

    def _validate_gann_rules(self, data, price, stop_loss=None, position_size=None):
        """Validate trade against Gann's 28 Trading Rules"""
        rules_validation = {
            'passed_rules': [],
            'failed_rules': [],
            'risk_score': 0,
            'discipline_score': 0,
            'structure_score': 0
        }
        
        # Calculate basic metrics
        current_price = data['Close'].iloc[-1]
        avg_volume = data['Volume'].mean()
        recent_volume = data['Volume'].tail(5).mean()
        atr = self._calculate_atr(data)
        
        # Rule 1: Use stops on every trade
        if stop_loss is not None:
            rules_validation['passed_rules'].append("Stop Loss Placed")
            rules_validation['discipline_score'] += 10
        else:
            rules_validation['failed_rules'].append("Missing Stop Loss")
        
        # Rule 2: Risk Management (10% max capital risk)
        if position_size is not None:
            risk_percent = ((price - stop_loss) / price * 100) if stop_loss else None
            if risk_percent and risk_percent <= 10:
                rules_validation['passed_rules'].append("Risk Within Limits")
                rules_validation['risk_score'] += 10
            else:
                rules_validation['failed_rules'].append("Risk Exceeds 10%")
        
        # Rule 3: Trend Following
        ema20 = data['Close'].ewm(span=20).mean()
        ema50 = data['Close'].ewm(span=50).mean()
        if current_price > ema20.iloc[-1] > ema50.iloc[-1]:
            rules_validation['passed_rules'].append("Following Uptrend")
            rules_validation['structure_score'] += 10
        elif current_price < ema20.iloc[-1] < ema50.iloc[-1]:
            rules_validation['passed_rules'].append("Following Downtrend")
            rules_validation['structure_score'] += 10
        
        # Rule 4: Volume Confirmation
        if recent_volume > avg_volume * 1.2:
            rules_validation['passed_rules'].append("Strong Volume")
            rules_validation['structure_score'] += 5
        
        # Rule 5: Support/Resistance Levels
        levels = self._identify_support_resistance(data)
        nearest_level = min(levels, key=lambda x: abs(x - current_price))
        if abs(current_price - nearest_level) / current_price < 0.02:
            rules_validation['passed_rules'].append("Near Key Level")
            rules_validation['structure_score'] += 5
        
        return rules_validation

    def _identify_gann_buy_sell_points(self, data):
        """Identify Gann's 12 specific entry/exit points"""
        signals = []
        current_price = data['Close'].iloc[-1]
        highs = data['High'].values
        lows = data['Low'].values
        closes = data['Close'].values
        
        # Pattern 1: Break of previous high after pullback
        if len(closes) > 20:
            recent_high = max(highs[-20:-1])
            recent_low = min(lows[-20:-1])
            if closes[-1] > recent_high and min(lows[-5:]) > recent_low:
                signals.append({
                    'type': 'BUY',
                    'pattern': 'Break of High After Pullback',
                    'strength': 'Strong',
                    'price': closes[-1]
                })
        
        # Pattern 2: Double Bottom
        if len(closes) > 40:
            lows_series = pd.Series(lows)
            potential_bottoms = lows_series[lows_series == lows_series.rolling(10).min()]
            if len(potential_bottoms) >= 2:
                last_two_bottoms = potential_bottoms.tail(2)
                if abs(last_two_bottoms.iloc[0] - last_two_bottoms.iloc[1]) / last_two_bottoms.iloc[0] < 0.02:
                    signals.append({
                        'type': 'BUY',
                        'pattern': 'Double Bottom',
                        'strength': 'Very Strong',
                        'price': closes[-1]
                    })
        
        # Pattern 3: Failed New High
        if len(closes) > 20:
            if max(highs[-5:]) < max(highs[-20:-5]) and closes[-1] < closes[-2]:
                signals.append({
                    'type': 'SELL',
                    'pattern': 'Failed New High',
                    'strength': 'Strong',
                    'price': closes[-1]
                })
        
        return signals

    def _analyze_monthly_cycles(self, data):
        """Analyze monthly high/low cycles for seasonal patterns"""
        monthly_data = data.resample('M').agg({
            'High': 'max',
            'Low': 'min',
            'Close': 'last'
        })
        
        cycle_analysis = {
            'strong_months': [],
            'weak_months': [],
            'current_month_bias': None,
            'historical_patterns': []
        }
        
        # Calculate monthly returns
        monthly_data['Returns'] = monthly_data['Close'].pct_change()
        
        # Analyze each month's historical performance
        for month in range(1, 13):
            month_data = monthly_data[monthly_data.index.month == month]
            avg_return = month_data['Returns'].mean()
            win_rate = (month_data['Returns'] > 0).mean()
            
            pattern = {
                'month': month,
                'avg_return': avg_return,
                'win_rate': win_rate,
                'significance': 'High' if abs(avg_return) > 0.02 else 'Moderate'
            }
            
            if avg_return > 0 and win_rate > 0.6:
                cycle_analysis['strong_months'].append(month)
            elif avg_return < 0 and win_rate < 0.4:
                cycle_analysis['weak_months'].append(month)
            
            cycle_analysis['historical_patterns'].append(pattern)
        
        # Analyze current month
        current_month = datetime.now().month
        if current_month in cycle_analysis['strong_months']:
            cycle_analysis['current_month_bias'] = 'Bullish'
        elif current_month in cycle_analysis['weak_months']:
            cycle_analysis['current_month_bias'] = 'Bearish'
        else:
            cycle_analysis['current_month_bias'] = 'Neutral'
        
        return cycle_analysis

    def _calculate_atr(self, data, period=14):
        """Calculate Average True Range"""
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr.iloc[-1]

    def _identify_support_resistance(self, data):
        """Identify key support and resistance levels"""
        levels = []
        
        # Use price action to find levels
        highs = data['High'].values
        lows = data['Low'].values
        
        # Find swing highs and lows
        for i in range(2, len(data)-2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
               highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                levels.append(highs[i])
            
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
               lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                levels.append(lows[i])
        
        return levels

    def generate_gann_analysis(self, symbol):
        """Enhanced generate_gann_analysis method with new features"""
        try:
            # Get stock data
            data = self.get_stock_data(symbol)
            if data is None or len(data) == 0:
                return None

            current_price = data['Close'].iloc[-1]
            
            # Calculate basic Gann indicators
            square_of_9 = self.calculate_square_of_9(current_price)
            angles = self.calculate_gann_angles(data)
            time_cycles = self.find_time_cycles(data)
            support_resistance = self.calculate_support_resistance(data, current_price)
            
            # Get enhanced pattern analysis
            gann_patterns = self.analyze_gann_patterns(data)
            time_price_balance = self.analyze_time_price_balance(data)
            
            # New Gann analysis features
            rules_validation = self._validate_gann_rules(
                data, 
                current_price,
                stop_loss=current_price * 0.95  # Example stop loss at 5% below current price
            )
            
            buy_sell_points = self._identify_gann_buy_sell_points(data)
            monthly_cycles = self._analyze_monthly_cycles(data)
            
            # Generate trading signals with enhanced analysis
            signals = self._generate_trading_signals(
                current_price, 
                square_of_9, 
                angles, 
                time_cycles, 
                gann_patterns,
                time_price_balance,
                rules_validation=rules_validation,
                buy_sell_points=buy_sell_points,
                monthly_cycles=monthly_cycles
            )
            
            # Determine exchange preference
            exchange_info = self._analyze_exchange_preference(symbol)
            
            return {
                'stock': symbol,
                'current_price': current_price,
                'recommendation': signals['recommendation'],
                'confidence': signals['confidence'],
                'trend': signals['trend'],
                'momentum': signals['momentum'],
                'suggested_time': signals['suggested_time'],
                'explanation': signals['explanation'],
                'support_levels': support_resistance['support_levels'],
                'resistance_levels': support_resistance['resistance_levels'],
                'time_cycles': time_cycles,
                'gann_patterns': gann_patterns,
                'time_price_balance': time_price_balance,
                'rules_validation': rules_validation,
                'buy_sell_points': buy_sell_points,
                'monthly_cycles': monthly_cycles,
                'exchange_preference': exchange_info,
                'scores': signals['scores']
            }
            
        except Exception as e:
            print(f"Error in generate_gann_analysis: {str(e)}")
            return None

    def _generate_trading_signals(self, price, square_of_9, angles, cycles, patterns, time_price_balance, rules_validation, buy_sell_points, monthly_cycles):
        """Generate trading signals based on comprehensive statistical analysis"""
        analysis_data = {
            'pattern_score': 0,
            'cycle_score': 0,
            'momentum_score': 0,
            'trend_score': 0,
            'volume_score': 0,
            'rules_score': 0,
            'signals': [],
            'confirmations': []
        }
        
        # Initialize recommendation parameters
        has_valid_pattern = False
        has_valid_cycle = False
        has_valid_momentum = False
        total_score = 0
        max_score = 0
        
        # 1. Pattern Analysis (20% weight)
        if patterns and isinstance(patterns, list):
            pattern_summary = patterns[-1] if patterns else None
            if pattern_summary and isinstance(pattern_summary, dict):
                very_strong = pattern_summary['strength_distribution'].get('Very Strong', 0)
                strong = pattern_summary['strength_distribution'].get('Strong', 0)
                moderate = pattern_summary['strength_distribution'].get('Moderate', 0)
                
                pattern_score = (very_strong * 3 + strong * 2 + moderate) / (pattern_summary['total_patterns'] or 1)
                analysis_data['pattern_score'] = min(pattern_score * 20, 20)  # Max 20 points
                has_valid_pattern = pattern_score > 0
                
                for pattern in patterns[:-1]:
                    if pattern['strength'] in ['Very Strong', 'Strong']:
                        analysis_data['signals'].append(f"{pattern['type']}: {pattern['name']} at ₹{pattern['price']:.2f}")
        
        # 2. Gann Rules Analysis (20% weight)
        if rules_validation:
            rules_score = (
                rules_validation['risk_score'] +
                rules_validation['discipline_score'] +
                rules_validation['structure_score']
            ) / 3
            analysis_data['rules_score'] = min(rules_score * 2, 20)  # Max 20 points
            
            # Add rule validations to confirmations
            for rule in rules_validation['passed_rules']:
                analysis_data['confirmations'].append(f"Rule Passed: {rule}")
        
        # 3. Buy/Sell Points Analysis (20% weight)
        if buy_sell_points:
            strong_signals = [s for s in buy_sell_points if s['strength'] in ['Very Strong', 'Strong']]
            buy_sell_score = len(strong_signals) * 5  # 5 points per strong signal
            analysis_data['buy_sell_score'] = min(buy_sell_score, 20)  # Max 20 points
            
            # Add buy/sell signals to confirmations
            for signal in strong_signals:
                analysis_data['signals'].append(
                    f"{signal['type']} Signal: {signal['pattern']} at ₹{signal['price']:.2f}"
                )
        
        # 4. Monthly Cycle Analysis (15% weight)
        if monthly_cycles:
            cycle_bias = monthly_cycles['current_month_bias']
            if cycle_bias == 'Bullish':
                analysis_data['cycle_score'] = 15
                analysis_data['confirmations'].append("Monthly Cycle: Bullish")
            elif cycle_bias == 'Bearish':
                analysis_data['cycle_score'] = 15
                analysis_data['confirmations'].append("Monthly Cycle: Bearish")
            else:
                analysis_data['cycle_score'] = 7.5
                analysis_data['confirmations'].append("Monthly Cycle: Neutral")
        
        # 5. Time-Price Balance (15% weight)
        if time_price_balance:
            strong_balance_points = [p for p in time_price_balance if p['significance'] in ['High', 'Very High']]
            balance_score = len(strong_balance_points) * 3  # 3 points per strong balance point
            analysis_data['balance_score'] = min(balance_score, 15)  # Max 15 points
            
            # Add strong balance points to confirmations
            for point in strong_balance_points[:2]:  # Show top 2 strongest points
                analysis_data['confirmations'].append(
                    f"Time-Price Balance: {point['type']} at ₹{point['price']:.2f}"
                )
        
        # 6. Traditional Cycle Analysis (10% weight)
        if cycles:
            strong_cycles = [corr for corr in cycles.values() if abs(corr) > 0.7]
            cycle_score = len(strong_cycles) * 2.5  # 2.5 points per strong cycle
            analysis_data['cycle_score'] = min(cycle_score, 10)  # Max 10 points
            
            # Add cycle confirmations
            for days, corr in cycles.items():
                if abs(corr) > 0.7:
                    analysis_data['confirmations'].append(
                        f"{days}-day cycle showing {abs(corr):.2f} correlation"
                    )
        
        # Calculate total score and confidence
        total_score = sum([
            analysis_data['pattern_score'],
            analysis_data['rules_score'],
            analysis_data.get('buy_sell_score', 0),
            analysis_data['cycle_score'],
            analysis_data.get('balance_score', 0)
        ])
        max_score = 100
        confidence = (total_score / max_score) * 100
        
        # Determine trend direction using multiple factors
        trend_factors = {
            'patterns': 'BULLISH' if analysis_data['pattern_score'] > 15 else 'BEARISH' if analysis_data['pattern_score'] < 5 else 'NEUTRAL',
            'rules': 'BULLISH' if analysis_data['rules_score'] > 15 else 'BEARISH' if analysis_data['rules_score'] < 5 else 'NEUTRAL',
            'cycles': monthly_cycles['current_month_bias'] if monthly_cycles else 'NEUTRAL'
        }
        
        # Count trend votes
        bullish_votes = sum(1 for v in trend_factors.values() if v == 'BULLISH')
        bearish_votes = sum(1 for v in trend_factors.values() if v == 'BEARISH')
        
        trend_direction = (
            'BULLISH' if bullish_votes > bearish_votes
            else 'BEARISH' if bearish_votes > bullish_votes
            else 'NEUTRAL'
        )
        
        # Generate recommendation based on comprehensive analysis
        if confidence >= 70 and has_valid_pattern and rules_validation['passed_rules']:
            if trend_direction == 'BULLISH' and monthly_cycles['current_month_bias'] != 'BEARISH':
                recommendation = 'STRONG BUY' if confidence > 85 else 'BUY'
            elif trend_direction == 'BEARISH' and monthly_cycles['current_month_bias'] != 'BULLISH':
                recommendation = 'STRONG SELL' if confidence > 85 else 'SELL'
            else:
                recommendation = 'HOLD'
                confidence = max(confidence * 0.8, 60)  # Reduce confidence for HOLD
        else:
            recommendation = 'NEUTRAL'
            confidence = max(confidence * 0.7, 50)  # Reduce confidence for NEUTRAL
        
        # Calculate target date based on strongest cycle or pattern
        target_date = None
        if monthly_cycles['current_month_bias'] != 'NEUTRAL':
            # Find next strong/weak month
            current_month = datetime.now().month
            if trend_direction == 'BULLISH':
                next_strong_months = [m for m in monthly_cycles['strong_months'] if m > current_month]
                if next_strong_months:
                    target_month = min(next_strong_months)
                    target_date = datetime(datetime.now().year, target_month, 1).strftime("%Y-%m-%d")
        else:
                next_weak_months = [m for m in monthly_cycles['weak_months'] if m > current_month]
                if next_weak_months:
                    target_month = min(next_weak_months)
                    target_date = datetime(datetime.now().year, target_month, 1).strftime("%Y-%m-%d")
        
        if not target_date and cycles:
            strongest_cycle = max(cycles.items(), key=lambda x: abs(x[1]))
            cycle_days = strongest_cycle[0]
            target_date = (datetime.now() + timedelta(days=int(cycle_days/2))).strftime("%Y-%m-%d")
        
        # Build detailed explanation
        explanation_parts = []
        
        # Add Gann rule validations
        if rules_validation['passed_rules']:
            explanation_parts.append("Passed Rules: " + "; ".join(rules_validation['passed_rules'][:3]))
        if rules_validation['failed_rules']:
            explanation_parts.append("Failed Rules: " + "; ".join(rules_validation['failed_rules']))
        
        # Add pattern signals
        if analysis_data['signals']:
            explanation_parts.append("Key Signals: " + "; ".join(analysis_data['signals'][:3]))
        
        # Add cycle confirmations
        if analysis_data['confirmations']:
            explanation_parts.append("Confirmations: " + "; ".join(analysis_data['confirmations'][:3]))
        
        # Add monthly cycle bias
        if monthly_cycles['current_month_bias'] != 'NEUTRAL':
            explanation_parts.append(f"Monthly Bias: {monthly_cycles['current_month_bias']}")
            
        return {
            'recommendation': recommendation,
            'confidence': round(confidence),
            'trend': trend_direction,
            'momentum': monthly_cycles['current_month_bias'],
            'suggested_time': target_date,
            'explanation': ' | '.join(explanation_parts) if explanation_parts else 'Insufficient data for analysis',
            'scores': analysis_data
        }

    def _calculate_rsi(self, data, period=14):
        """Calculate Relative Strength Index"""
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs)).iloc[-1]

    def _calculate_macd(self, data, fast=12, slow=26, signal=9):
        """Calculate MACD and Signal line"""
        exp1 = data['Close'].ewm(span=fast, adjust=False).mean()
        exp2 = data['Close'].ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd.iloc[-1], signal_line.iloc[-1]

    def _calculate_stochastic(self, data, period=14, k=3, d=3):
        """Calculate Stochastic Oscillator"""
        low_min = data['Low'].rolling(window=period).min()
        high_max = data['High'].rolling(window=period).max()
        k_line = ((data['Close'] - low_min) / (high_max - low_min)) * 100
        k_smooth = k_line.rolling(window=k).mean()
        d_line = k_smooth.rolling(window=d).mean()
        return k_smooth.iloc[-1], d_line.iloc[-1]

    def _analyze_trend_strength(self, data):
        """Analyze trend strength using multiple indicators"""
        # Calculate EMAs
        ema20 = data['Close'].ewm(span=20, adjust=False).mean()
        ema50 = data['Close'].ewm(span=50, adjust=False).mean()
        ema200 = data['Close'].ewm(span=200, adjust=False).mean()
        
        # Calculate ADX for trend strength
        plus_dm = data['High'].diff()
        minus_dm = data['Low'].diff()
        tr = pd.DataFrame(
            [data['High'] - data['Low'],
             abs(data['High'] - data['Close'].shift(1)),
             abs(data['Low'] - data['Close'].shift(1))]).max()
        plus_di = 100 * (plus_dm.where(plus_dm > 0, 0).ewm(span=14).mean() / tr.ewm(span=14).mean())
        minus_di = 100 * (minus_dm.where(minus_dm > 0, 0).ewm(span=14).mean() / tr.ewm(span=14).mean())
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(span=14).mean().iloc[-1]
        
        # Score trend strength (0-20 points)
        score = 0
        
        # EMA alignments (max 8 points)
        if ema20.iloc[-1] > ema50.iloc[-1] > ema200.iloc[-1]:
            score += 8  # Strong uptrend
        elif ema20.iloc[-1] < ema50.iloc[-1] < ema200.iloc[-1]:
            score -= 8  # Strong downtrend
        elif ema20.iloc[-1] > ema50.iloc[-1]:
            score += 4  # Moderate uptrend
        elif ema20.iloc[-1] < ema50.iloc[-1]:
            score -= 4  # Moderate downtrend
        
        # ADX strength (max 6 points)
        if adx > 30:
            score += 6  # Strong trend
        elif adx > 20:
            score += 3  # Moderate trend
        
        # Price position relative to EMAs (max 6 points)
        current_price = data['Close'].iloc[-1]
        if current_price > ema20.iloc[-1]:
            score += 2
        if current_price > ema50.iloc[-1]:
            score += 2
        if current_price > ema200.iloc[-1]:
            score += 2
        
        return abs(score)

    def _analyze_volume_strength(self, data):
        """Analyze volume strength and trends"""
        # Calculate volume metrics
        avg_volume = data['Volume'].mean()
        recent_volume = data['Volume'].tail(5).mean()
        volume_trend = recent_volume / avg_volume
        
        # Calculate On-Balance Volume (OBV)
        obv = (data['Volume'] * (~data['Close'].diff().le(0) * 2 - 1)).cumsum()
        obv_trend = obv.diff(5).iloc[-1] > 0
        
        # Score volume strength (0-15 points)
        score = 0
        
        # Volume level (max 5 points)
        if volume_trend > 1.5:
            score += 5  # Strong volume
        elif volume_trend > 1.0:
            score += 3  # Moderate volume
        elif volume_trend > 0.7:
            score += 1  # Weak volume
        
        # Volume trend (max 5 points)
        if obv_trend:
            score += 5  # Positive OBV trend
        
        # Volume consistency (max 5 points)
        volume_std = data['Volume'].tail(10).std() / avg_volume
        if volume_std < 0.5:
            score += 5  # Consistent volume
        elif volume_std < 1.0:
            score += 3  # Moderately consistent
        elif volume_std < 1.5:
            score += 1  # Volatile volume
        
        return score

    def _analyze_exchange_preference(self, symbol):
        """Analyze which exchange is preferred for a given symbol"""
        try:
            # Try both exchanges
            nse_data = None
            bse_data = None
            
            # Try NSE
            try:
                nse_symbol = symbol + '.NS'
                stock = yf.Ticker(nse_symbol)
                nse_data = stock.history(period='1mo')
                nse_volume = nse_data['Volume'].mean() if not nse_data.empty else 0
            except:
                nse_volume = 0
            
            # Try BSE
            try:
                bse_symbol = symbol + '.BO'
                stock = yf.Ticker(bse_symbol)
                bse_data = stock.history(period='1mo')
                bse_volume = bse_data['Volume'].mean() if not bse_data.empty else 0
            except:
                bse_volume = 0
            
            # Compare volumes and availability
            if nse_volume > 0 and bse_volume > 0:
                if nse_volume > bse_volume:
                    return 'NSE', nse_volume / bse_volume
                else:
                    return 'BSE', bse_volume / nse_volume
            elif nse_volume > 0:
                return 'NSE', float('inf')
            elif bse_volume > 0:
                return 'BSE', float('inf')
            
            return None, 0
            
        except Exception as e:
            print(f"Error analyzing exchange preference: {str(e)}")
            return None, 0