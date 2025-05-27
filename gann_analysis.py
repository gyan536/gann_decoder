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
        """Create comprehensive symbol mapping from CSV file"""
        try:
            # Try to read the CSV file
            csv_path = os.path.join(os.path.dirname(__file__), 'indian_stocks.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                # Create mapping dictionary
                mapping = {}
                for _, row in df.iterrows():
                    mapping[row['Symbol']] = row['Trading_Symbol']
                print(f"Loaded {len(mapping)} stock mappings from CSV")
                return mapping
            
            # If CSV doesn't exist, use default mappings
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
        """Format and clean the input symbol"""
        # Remove common suffixes and clean the input
        clean_symbol = input_symbol.upper().strip()
        clean_symbol = clean_symbol.replace('.NS', '').replace('.BO', '')
        clean_symbol = clean_symbol.replace('NSE:', '').replace('BSE:', '')
        
        # Try direct match in cache
        if clean_symbol in self.symbols_cache:
            return self.symbols_cache[clean_symbol]
        
        # Try without common suffixes
        common_suffixes = [' LTD', ' LIMITED', ' LTD.', ' LIMITED.', ' INDIA', 
                         ' INDIA LIMITED', ' CORPORATION', ' CORP']
        test_symbol = clean_symbol
        for suffix in common_suffixes:
            if test_symbol.endswith(suffix):
                test_symbol = test_symbol[:-len(suffix)]
                if test_symbol in self.symbols_cache:
                    return self.symbols_cache[test_symbol]
        
        # Try partial matches
        for company in self.symbols_cache:
            if (clean_symbol in company or company in clean_symbol) and \
               len(clean_symbol) > 3 and len(company) > 3:
                return self.symbols_cache[company]
        
        # If no match found, clean up and return the original
        return ''.join(e for e in clean_symbol if e.isalnum())

    def get_stock_data(self, symbol, period='1y'):
        """Fetch stock data from multiple sources with improved error handling"""
        # Format the symbol
        clean_symbol = self._format_symbol(symbol)
        errors = []
        
        print(f"Processing request for: {symbol}")
        print(f"Mapped to symbol: {clean_symbol}")
        
        # Try NSE first
        try:
            print(f"Attempting NSE data fetch for: {clean_symbol}")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            
            try:
                data = get_history(symbol=clean_symbol, start=start_date, end=end_date)
                if not data.empty and len(data) > 0:
                    print(f"Successfully fetched NSE data for: {clean_symbol}")
                    return data
            except Exception as nse_error:
                errors.append(f"NSE: {str(nse_error)}")
                print(f"NSE fetch failed, trying Yahoo Finance...")
        except Exception as e:
            errors.append(f"NSE setup: {str(e)}")
        
        # Try Yahoo Finance with multiple formats
        try:
            symbol_formats = [
                clean_symbol + '.NS',  # NSE format
                clean_symbol + '.BO',  # BSE format
                'NSE:' + clean_symbol, # Alternative NSE format
                'BSE:' + clean_symbol, # Alternative BSE format
                clean_symbol           # Raw symbol
            ]
            
            for sym in symbol_formats:
                try:
                    print(f"Attempting to fetch data for: {sym}")
                    stock = yf.Ticker(sym)
                    data = stock.history(period=period)
                    if not data.empty and len(data) > 0:
                        print(f"Successfully fetched data for: {sym}")
                        return data
                except Exception as e:
                    errors.append(f"Yahoo Finance ({sym}): {str(e)}")
                    continue
        except Exception as e:
            errors.append(f"Yahoo Finance setup: {str(e)}")
        
        # If all attempts fail, raise an error with details
        error_msg = (f"Could not fetch data for '{symbol}' (mapped to '{clean_symbol}'). "
                    f"Please verify the company name or stock symbol. Tried:\n")
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

    def generate_gann_analysis(self, symbol):
        """Generate comprehensive Gann analysis"""
        try:
            # Get stock data
            data = self.get_stock_data(symbol)
            if data.empty:
                raise Exception("No data available for analysis")

            current_price = data['Close'].iloc[-1]
            
            # Perform various Gann analyses
            square_of_9 = self.calculate_square_of_9(current_price)
            gann_angles = self.calculate_gann_angles(data)
            time_cycles = self.find_time_cycles(data)
            support_resistance = self.calculate_support_resistance(data, current_price)
            price_patterns = self.analyze_price_patterns(data)
            
            # Generate trading signals
            signals = self._generate_trading_signals(
                current_price, 
                square_of_9, 
                gann_angles, 
                time_cycles, 
                price_patterns
            )
            
            return {
                'status': 'SUCCESS',
                'stock': symbol,
                'current_price': round(current_price, 2),
                'recommendation': signals['action'],
                'confidence': signals['confidence'],
                'explanation': signals['explanation'],
                'suggested_time': signals['timing'],
                'support_levels': support_resistance['support_levels'],
                'resistance_levels': support_resistance['resistance_levels'],
                'trend': price_patterns['trend'],
                'momentum': price_patterns['momentum'],
                'time_cycles': time_cycles
            }
            
        except Exception as e:
            return {
                'status': 'ERROR',
                'stock': symbol,
                'message': str(e)
            }

    def _generate_trading_signals(self, price, square_of_9, angles, cycles, patterns):
        """Generate trading signals based on Gann analysis"""
        # Initialize scoring system
        buy_score = 0
        sell_score = 0
        signals = []
        
        # Check price position relative to Square of 9 levels
        nearest_level = min(square_of_9, key=lambda x: abs(x - price))
        price_level_diff = ((price - nearest_level) / price) * 100
        
        if abs(price_level_diff) < 1:  # Price near Gann level
            signals.append("Price at critical Gann Square of 9 level")
            if price > nearest_level:
                buy_score += 2
            else:
                sell_score += 2
        
        # Analyze trend and momentum
        if patterns['trend'] == 'BULLISH':
            buy_score += 2
            signals.append("Bullish trend on Gann angles")
        else:
            sell_score += 2
            signals.append("Bearish trend on Gann angles")
            
        if patterns['momentum'] == 'OVERBOUGHT':
            sell_score += 1
            signals.append("Overbought on momentum indicators")
        elif patterns['momentum'] == 'OVERSOLD':
            buy_score += 1
            signals.append("Oversold on momentum indicators")
            
        # Check time cycles
        strongest_cycle = max(cycles.items(), key=lambda x: abs(x[1])) if cycles else None
        if strongest_cycle and abs(strongest_cycle[1]) > 0.7:
            signals.append(f"Strong {strongest_cycle[0]}-day cycle detected")
            if strongest_cycle[1] > 0:
                buy_score += 1
            else:
                sell_score += 1
        
        # Generate final recommendation
        total_score = buy_score + sell_score
        confidence = min((max(buy_score, sell_score) / total_score * 100) if total_score > 0 else 50, 95)
        
        if buy_score > sell_score:
            action = 'STRONG BUY' if buy_score > sell_score + 2 else 'BUY'
        elif sell_score > buy_score:
            action = 'STRONG SELL' if sell_score > buy_score + 2 else 'SELL'
        else:
            action = 'HOLD'
            confidence = 50
        
        # Generate timing based on strongest cycle
        if strongest_cycle:
            days_ahead = min(strongest_cycle[0] // 4, 30)  # Cap at 30 days
            suggested_time = datetime.now() + timedelta(days=days_ahead)
        else:
            suggested_time = datetime.now() + timedelta(days=7)  # Default to 7 days
            
        return {
            'action': action,
            'confidence': round(confidence),
            'explanation': ' | '.join(signals),
            'timing': suggested_time.strftime("%Y-%m-%d")
        } 