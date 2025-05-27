# Gann Market Analysis with ML

A comprehensive market analysis system combining traditional Gann theory with modern machine learning capabilities.

## Features

- Traditional Gann Analysis
  - Square of 9 calculations
  - Support and resistance levels
  - Time cycles analysis
  - Price pattern recognition

- ML-Enhanced Analysis
  - LSTM for price prediction
  - Random Forest for pattern detection
  - XGBoost for trend prediction
  - Isolation Forest for anomaly detection

- Combined Analysis Features
  - Market wave identification
  - Risk level assessment
  - Trend strength analysis
  - Trading signals generation

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd gann_decoder_ai
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. Enter a stock symbol (e.g., "HDFCBANK.NS" for HDFC Bank NSE listing)

4. Choose analysis type:
   - Traditional: Gann analysis only
   - ML: Machine learning analysis only
   - Both: Combined analysis

## Stock Symbol Format

- Indian NSE stocks: Add ".NS" (e.g., "RELIANCE.NS")
- Indian BSE stocks: Add ".BO" (e.g., "RELIANCE.BO")
- US stocks: Use symbol directly (e.g., "AAPL")

## API Endpoints

- `/`: Main page
- `/analyze`: POST endpoint for analysis
- `/api/stocks`: GET endpoint for stock symbol search
- `/api/refresh-analysis`: POST endpoint to refresh analysis

## File Structure

- `app.py`: Main Flask application
- `gann_logic.py`: Traditional Gann analysis
- `ml_market_analysis.py`: ML-based market analysis
- `templates/`: HTML templates
  - `index.html`: Input form
  - `result.html`: Analysis results display

## Error Handling

The system includes robust error handling for:
- Invalid stock symbols
- Insufficient data
- API failures
- Data validation
- ML model errors

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request
