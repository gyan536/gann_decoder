# GANN Decoder AI

A sophisticated stock market analysis tool that combines GANN analysis techniques with machine learning to provide market insights and predictions.

## Features

- GANN analysis for stock market patterns
- Machine learning-based market predictions
- Real-time stock data integration with Yahoo Finance
- Interactive web interface
- Support for both Indian (NSE/BSE) and US markets
- Technical indicators and pattern recognition
- Time cycle analysis

## Project Structure

```
gann_decoder_ai/
├── app/                    # Application package
│   ├── __init__.py
│   ├── api/               # API endpoints
│   ├── core/              # Core business logic
│   ├── models/            # Data models
│   └── utils/             # Utility functions
├── config/                # Configuration files
├── data/                  # Data storage
├── static/               # Static files (CSS, JS)
├── templates/            # HTML templates
├── tests/                # Test suite
└── logs/                 # Application logs
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/gann_decoder_ai.git
cd gann_decoder_ai
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up configuration:
```bash
cp config/config.example.py config/config.py
# Edit config.py with your settings
```

## Usage

1. Start the application:
```bash
python run.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

## Configuration

The application can be configured through:
- Environment variables
- Configuration file (`config/config.py`)
- Command line arguments

Key configuration options:
- `FLASK_ENV`: Development/Production environment
- `DEBUG`: Debug mode
- `SECRET_KEY`: Application secret key
- `DATABASE_URL`: Database connection string
- `CACHE_TYPE`: Caching backend
- `LOG_LEVEL`: Logging level

## API Documentation

The application provides a RESTful API for stock analysis:

### Endpoints

- `GET /api/stocks`: Search available stocks
- `POST /api/analyze`: Perform stock analysis
- `POST /api/refresh-analysis`: Refresh analysis for a stock

Detailed API documentation is available at `/api/docs` when running the application.

## Development

### Setting up development environment

1. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

2. Set up pre-commit hooks:
```bash
pre-commit install
```

### Running tests

```bash
pytest
```

### Code style

This project follows PEP 8 style guide. Use flake8 and black for code formatting:

```bash
black .
flake8
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- GANN analysis techniques
- Machine learning libraries contributors
- Yahoo Finance API
- Flask framework
