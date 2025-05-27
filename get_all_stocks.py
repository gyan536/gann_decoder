import pandas as pd
import requests
import json
from bs4 import BeautifulSoup
import os
import time
from datetime import datetime
from io import StringIO  # Fix for StringIO import
import yfinance as yf
from fuzzywuzzy import fuzz

def fetch_nse_stocks():
    """Fetch list of stocks from NSE"""
    print("Fetching NSE stocks...")
    try:
        # NSE stocks list URL
        url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }

        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text))
            df = df[['SYMBOL', 'NAME OF COMPANY']]
            df.columns = ['Symbol', 'Company_Name']
            df['Exchange'] = 'NSE'
            print(f"Successfully fetched {len(df)} NSE stocks")
            return df
        else:
            print(f"Failed to fetch NSE stocks: Status code {response.status_code}")
            return pd.DataFrame()
    except Exception as e:
        print(f"Error fetching NSE stocks: {str(e)}")
        return pd.DataFrame()

def fetch_bse_stocks():
    """Fetch list of stocks from BSE"""
    print("Fetching BSE stocks...")
    try:
        # BSE stocks list URL
        url = "https://www.bseindia.com/corporates/List_Scrips.aspx"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        stocks = []
        for row in soup.select('#ContentPlaceHolder1_gvData tr')[1:]:  # Skip header row
            cols = row.select('td')
            if len(cols) >= 2:
                symbol = cols[0].text.strip()
                name = cols[1].text.strip()
                stocks.append({
                    'Symbol': symbol,
                    'Company_Name': name,
                    'Exchange': 'BSE'
                })
        
        df = pd.DataFrame(stocks)
        print(f"Successfully fetched {len(df)} BSE stocks")
        return df
    except Exception as e:
        print(f"Error fetching BSE stocks: {str(e)}")
        return pd.DataFrame()

def generate_symbol_variations(row):
    """Generate variations of company symbols and names"""
    variations = []
    symbol = row['Symbol'].strip().upper()
    name = row['Company_Name'].strip().upper()
    exchange = row['Exchange']
    
    # Add original
    variations.append({
        'Symbol': symbol,
        'Trading_Symbol': symbol,
        'Company_Name': name,
        'Exchange': exchange
    })
    
    # Add with exchange suffix
    if exchange == 'NSE':
        variations.append({
            'Symbol': f"{symbol}.NS",
            'Trading_Symbol': symbol,
            'Company_Name': name,
            'Exchange': exchange
        })
    elif exchange == 'BSE':
        variations.append({
            'Symbol': f"{symbol}.BO",
            'Trading_Symbol': symbol,
            'Company_Name': name,
            'Exchange': exchange
        })
    
    # Clean and standardize company name
    clean_name = name.strip().upper()
    name_variations = [
        clean_name,
        clean_name + " LIMITED",
        clean_name + " LTD",
        clean_name.replace("LIMITED", "").strip(),
        clean_name.replace("LTD", "").strip(),
        clean_name.replace("AND", "&").strip(),
        clean_name.replace("&", "AND").strip(),
        ''.join(e for e in clean_name if e.isalnum())  # Alphanumeric only
    ]
    
    # Add name variations
    for var in name_variations:
        if var:  # Only add non-empty variations
            variations.append({
                'Symbol': var,
                'Trading_Symbol': symbol,
                'Company_Name': name,
                'Exchange': exchange
            })
    
    return variations

def create_stock_list():
    """Create comprehensive stock list from both NSE and BSE"""
    # Create output directory if it doesn't exist
    output_dir = 'data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Fetch stocks from both exchanges
    nse_stocks = fetch_nse_stocks()
    time.sleep(1)  # Add delay to avoid rate limiting
    bse_stocks = fetch_bse_stocks()
    
    # Combine stocks from both exchanges
    all_stocks = pd.concat([nse_stocks, bse_stocks], ignore_index=True)
    
    if all_stocks.empty:
        print("Error: Could not fetch stocks from either exchange")
        return
    
    # Generate variations for each stock
    variations = []
    for _, row in all_stocks.iterrows():
        variations.extend(generate_symbol_variations(row))
    
    # Convert to DataFrame and remove duplicates
    variations_df = pd.DataFrame(variations)
    variations_df = variations_df.drop_duplicates(subset=['Symbol'])
    
    # Save original stock list
    timestamp = datetime.now().strftime("%Y%m%d")
    all_stocks.to_csv(f"{output_dir}/stock_list_{timestamp}.csv", index=False)
    print(f"Saved {len(all_stocks)} stocks to stock_list_{timestamp}.csv")
    
    # Save variations with trading symbols
    variations_df.to_csv(f"{output_dir}/stock_variations_{timestamp}.csv", index=False)
    print(f"Saved {len(variations_df)} stock variations to stock_variations_{timestamp}.csv")
    
    # Save as JSON for quick loading
    variations_dict = variations_df.set_index('Symbol')['Trading_Symbol'].to_dict()
    with open(f"{output_dir}/stock_symbols.json", 'w') as f:
        json.dump(variations_dict, f, indent=2)
    print(f"Saved stock symbols mapping to stock_symbols.json")

class StockSymbolManager:
    def __init__(self):
        self.cache_file = 'stock_symbols_cache.json'
        self.cache_expiry_days = 1  # Cache expires after 1 day
        self.symbols_data = self._load_cached_data()

    def _load_cached_data(self):
        """Load stock symbols from cache if available and not expired"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    cache = json.load(f)
                    # Check if cache is expired
                    last_updated = datetime.fromisoformat(cache['last_updated'])
                    if (datetime.now() - last_updated).days < self.cache_expiry_days:
                        return cache['data']
            except Exception as e:
                print(f"Error loading cache: {str(e)}")
        return self._fetch_and_cache_data()

    def _fetch_and_cache_data(self):
        """Fetch stock symbols from various exchanges and cache them"""
        symbols_data = []

        try:
            # NSE stocks list URL
            url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                # Read CSV directly from response content
                df = pd.read_csv(StringIO(response.text))
                
                # Process NSE stocks
                for _, row in df.iterrows():
                    symbol = str(row['SYMBOL']).strip()
                    name = str(row['NAME OF COMPANY']).strip()
                    
                    # Add NSE symbol
                    symbols_data.append({
                        'symbol': f"{symbol}.NS",
                        'name': name,
                        'exchange': 'NSE'
                    })
                    
                    # Add BSE symbol variant
                    symbols_data.append({
                        'symbol': f"{symbol}.BO",
                        'name': name,
                        'exchange': 'BSE'
                    })
                    
                    # Add symbol without exchange suffix
                    symbols_data.append({
                        'symbol': symbol,
                        'name': name,
                        'exchange': 'NSE'
                    })
            else:
                print(f"Failed to fetch NSE data: Status code {response.status_code}")
                # Use backup data if available
                if os.path.exists('data/stock_list.csv'):
                    df = pd.read_csv('data/stock_list.csv')
                    for _, row in df.iterrows():
                        symbols_data.append({
                            'symbol': row['Symbol'],
                            'name': row['Company_Name'],
                            'exchange': row['Exchange']
                        })

        except Exception as e:
            print(f"Error fetching stock data: {str(e)}")
            # Use backup data if available
            if os.path.exists('data/stock_list.csv'):
                df = pd.read_csv('data/stock_list.csv')
                for _, row in df.iterrows():
                    symbols_data.append({
                        'symbol': row['Symbol'],
                        'name': row['Company_Name'],
                        'exchange': row['Exchange']
                    })

        if not symbols_data:
            # Add some common stock symbols as fallback
            fallback_symbols = [
                ('HDFCBANK', 'HDFC Bank Ltd', 'NSE'),
                ('RELIANCE', 'Reliance Industries Ltd', 'NSE'),
                ('TCS', 'Tata Consultancy Services Ltd', 'NSE'),
                ('INFY', 'Infosys Ltd', 'NSE'),
                ('TATAMOTORS', 'Tata Motors Ltd', 'NSE')
            ]
            
            for symbol, name, exchange in fallback_symbols:
                # Add with exchange suffix
                symbols_data.append({
                    'symbol': f"{symbol}.NS",
                    'name': name,
                    'exchange': 'NSE'
                })
                symbols_data.append({
                    'symbol': f"{symbol}.BO",
                    'name': name,
                    'exchange': 'BSE'
                })
                # Add without suffix
                symbols_data.append({
                    'symbol': symbol,
                    'name': name,
                    'exchange': exchange
                })

        # Cache the data
        cache_data = {
            'last_updated': datetime.now().isoformat(),
            'data': symbols_data
        }
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f)
        except Exception as e:
            print(f"Error caching data: {str(e)}")

        return symbols_data

    def search_symbol(self, query):
        """Search for stock symbols using fuzzy matching"""
        query = query.upper()
        matches = []
        
        for stock in self.symbols_data:
            symbol_score = fuzz.ratio(query, stock['symbol'].split('.')[0])
            name_score = fuzz.partial_ratio(query, stock['name'].upper())
            
            # Use the higher of the two scores
            score = max(symbol_score, name_score)
            
            if score > 60:  # Threshold for matches
                matches.append({
                    'symbol': stock['symbol'],
                    'name': stock['name'],
                    'exchange': stock['exchange'],
                    'score': score
                })
        
        # Sort by score and return top 10 matches
        return sorted(matches, key=lambda x: x['score'], reverse=True)[:10]

    def validate_symbol(self, symbol):
        """Validate a stock symbol and suggest alternatives if needed"""
        # Check if symbol exists exactly as provided
        exact_match = next((s for s in self.symbols_data if s['symbol'].upper() == symbol.upper()), None)
        if exact_match:
            return {'valid': True, 'symbol': exact_match['symbol']}

        # If symbol has exchange suffix, try without it
        if '.' in symbol:
            base_symbol = symbol.split('.')[0]
            matches = [s for s in self.symbols_data if s['symbol'].startswith(f"{base_symbol}.")]
            if matches:
                return {
                    'valid': False,
                    'message': f"Could not find valid symbol for '{symbol}'. Try these alternatives:",
                    'suggestions': [{'symbol': m['symbol'], 'name': m['name']} for m in matches[:3]]
                }

        # Try fuzzy matching for suggestions
        matches = self.search_symbol(symbol)
        if matches:
            return {
                'valid': False,
                'message': f"Could not find exact match for '{symbol}'. Did you mean:",
                'suggestions': [{'symbol': m['symbol'], 'name': m['name']} for m in matches[:3]]
            }

        return {
            'valid': False,
            'message': f"Could not find any matching symbols for '{symbol}'.",
            'suggestions': []
        }

# Global instance
stock_manager = StockSymbolManager()

if __name__ == "__main__":
    create_stock_list()

    # Test the functionality
    test_symbols = ['HDFCBANK.NS', 'RELIANCE.BO', 'INFY', 'TATAMOTORS']
    for symbol in test_symbols:
        result = stock_manager.validate_symbol(symbol)
        print(f"\nValidating {symbol}:")
        print(json.dumps(result, indent=2))

        print(f"\nSearching for {symbol}:")
        matches = stock_manager.search_symbol(symbol)
        print(json.dumps(matches, indent=2)) 