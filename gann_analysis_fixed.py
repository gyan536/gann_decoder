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