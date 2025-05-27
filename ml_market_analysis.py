import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Union
from dataclasses import dataclass
from enum import Enum
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
import xgboost as xgb
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import torch
import torch.nn as nn
from datetime import datetime, timedelta

class TrendDirection(Enum):
    UP = "UP"
    DOWN = "DOWN"
    SIDEWAYS = "SIDEWAYS"

@dataclass
class MarketWave:
    start_price: float
    end_price: float
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    direction: TrendDirection
    duration_days: float
    confidence: float = 1.0

class LSTMPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int):
        super(LSTMPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out

    def identify_market_waves_ml(self) -> List[MarketWave]:
        """
        Identify market waves using ML techniques
        """
        # Get pattern probabilities
        pattern_probs = self.pattern_detector.predict_proba(self.df[self.features].values)
        
        # Get anomaly scores for potential swing points
        anomaly_scores = self.anomaly_detector.score_samples(self.df[self.features].values)
        
        waves = []
        current_wave_start = 0
        current_direction = None
        
        for i in range(1, len(self.df)):
            # Combine ML signals
            is_swing_point = anomaly_scores[i] < np.percentile(anomaly_scores, 10)
            pattern_change = np.argmax(pattern_probs[i]) != np.argmax(pattern_probs[i-1])
            
            if is_swing_point or pattern_change:
                if current_direction is not None:
                    # Create wave
                    wave = MarketWave(
                        start_price=self.df['Close'].iloc[current_wave_start],
                        end_price=self.df['Close'].iloc[i],
                        start_time=self.df.index[current_wave_start],
                        end_time=self.df.index[i],
                        direction=current_direction,
                        duration_days=(self.df.index[i] - self.df.index[current_wave_start]).days,
                        confidence=abs(anomaly_scores[i])  # Use anomaly score as confidence
                    )
                    waves.append(wave)
                
                # Start new wave
                current_wave_start = i
                price_change = self.df['Close'].iloc[i] - self.df['Close'].iloc[i-1]
                current_direction = TrendDirection.UP if price_change > 0 else TrendDirection.DOWN
        
        return waves

    def predict_next_wave(self) -> Dict:
        """
        Predict characteristics of the next market wave
        """
        recent_data = self.df.tail(10)[self.features].values
        
        # Predict trend
        trend_pred = self.trend_predictor.predict(recent_data[-1:])
        trend_probs = self.trend_predictor.predict_proba(recent_data[-1:])
        
        # Prepare sequence for LSTM
        sequence = torch.FloatTensor(recent_data.reshape(1, 10, -1))
        
        # Predict next price
        self.lstm_predictor.eval()
        with torch.no_grad():
            next_price_scaled = self.lstm_predictor(sequence)
            next_price = self.scaler.inverse_transform(
                next_price_scaled.numpy().reshape(1, -1)
            )[0, 0]
        
        return {
            "predicted_trend": TrendDirection(trend_pred[0]),
            "confidence": float(np.max(trend_probs)),
            "predicted_price": float(next_price)
        }

class MLMarketAnalyzer:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with OHLC DataFrame with ML capabilities
        df should have columns: Open, High, Low, Close, Date/Timestamp
        """
        self.df = df
        self.df['Timestamp'] = pd.to_datetime(self.df.index)
        self.zigzag_threshold = 0.05
        self.scaler = MinMaxScaler()
        self._prepare_features()
        self._initialize_models()

    def _prepare_features(self):
        """
        Prepare technical indicators and features for ML models
        """
        # Add technical indicators
        self.df['SMA20'] = SMAIndicator(close=self.df['Close'], window=20).sma_indicator()
        self.df['EMA20'] = EMAIndicator(close=self.df['Close'], window=20).ema_indicator()
        self.df['RSI'] = RSIIndicator(close=self.df['Close']).rsi()
        
        bb = BollingerBands(close=self.df['Close'])
        self.df['BB_upper'] = bb.bollinger_hband()
        self.df['BB_lower'] = bb.bollinger_lband()
        
        # Calculate returns and volatility
        self.df['Returns'] = self.df['Close'].pct_change()
        self.df['Volatility'] = self.df['Returns'].rolling(window=20).std()
        
        # Create features for ML models
        self.features = ['Open', 'High', 'Low', 'Close', 'SMA20', 'EMA20', 'RSI', 
                        'BB_upper', 'BB_lower', 'Returns', 'Volatility']
        
        self.df = self.df.dropna()

    def _initialize_models(self):
        """
        Initialize ML models for different analysis tasks
        """
        # Pattern detection model (Random Forest)
        self.pattern_detector = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Anomaly detection for swing points
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        
        # Trend prediction model (XGBoost)
        self.trend_predictor = xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=3,
            random_state=42
        )
        
        # Price prediction model (LSTM)
        self.lstm_predictor = LSTMPredictor(
            input_dim=len(self.features),
            hidden_dim=64,
            num_layers=2,
            output_dim=1
        )

    def _prepare_sequence_data(self, sequence_length: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequence data for LSTM model
        """
        data = self.df[self.features].values
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(len(scaled_data) - sequence_length):
            X.append(scaled_data[i:(i + sequence_length)])
            y.append(scaled_data[i + sequence_length, 3])  # Predict Close price
            
        return np.array(X), np.array(y)

    def train_models(self):
        """
        Train all ML models
        """
        # Prepare data for pattern detection
        X = self.df[self.features].values
        y = np.where(self.df['Returns'] > 0, 1, 0)  # Simple binary classification
        
        # Train pattern detector
        self.pattern_detector.fit(X, y)
        
        # Train anomaly detector
        self.anomaly_detector.fit(X)
        
        # Train trend predictor
        trend_labels = np.where(self.df['Returns'] > 0.02, 2,  # Strong UP
                              np.where(self.df['Returns'] < -0.02, 0, 1))  # Strong DOWN, else SIDEWAYS
        self.trend_predictor.fit(X, trend_labels)
        
        # Train LSTM predictor
        X_seq, y_seq = self._prepare_sequence_data()
        X_tensor = torch.FloatTensor(X_seq)
        y_tensor = torch.FloatTensor(y_seq)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.lstm_predictor.parameters())
        
        # Train LSTM
        self.lstm_predictor.train()
        for epoch in range(50):
            optimizer.zero_grad()
            output = self.lstm_predictor(X_tensor)
            loss = criterion(output.squeeze(), y_tensor)
            loss.backward()
            optimizer.step()

    def analyze_market_state(self) -> Dict:
        """
        Comprehensive market analysis using ML
        """
        waves = self.identify_market_waves_ml()
        next_wave_pred = self.predict_next_wave()
        
        current_price = self.df['Close'].iloc[-1]
        
        analysis = {
            "current_state": {
                "price": current_price,
                "trend": self.get_current_trend(),
                "volatility": self.df['Volatility'].iloc[-1]
            },
            "wave_analysis": {
                "total_waves": len(waves),
                "average_duration": np.mean([w.duration_days for w in waves]),
                "average_confidence": np.mean([w.confidence for w in waves])
            },
            "prediction": next_wave_pred,
            "risk_level": self.calculate_risk_level()
        }
        
        return analysis

    def get_current_trend(self) -> TrendDirection:
        """
        Get current trend using ML model
        """
        recent_data = self.df.tail(1)[self.features].values
        predicted_trend = self.trend_predictor.predict(recent_data)[0]
        return TrendDirection(predicted_trend)

    def calculate_risk_level(self) -> float:
        """
        Calculate current market risk level using ML
        """
        recent_data = self.df.tail(20)[self.features].values
        anomaly_scores = self.anomaly_detector.score_samples(recent_data)
        risk_level = 1 - np.mean(anomaly_scores)  # Higher anomaly score = lower risk
        return risk_level 