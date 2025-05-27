import pandas as pd
import requests
import json
from bs4 import BeautifulSoup
import os
import time
from datetime import datetime
from io import StringIO  # Fix for StringIO import

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

if __name__ == "__main__":
    create_stock_list() 