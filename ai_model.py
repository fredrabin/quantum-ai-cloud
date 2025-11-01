import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib
import ta
from ta import add_all_ta_features
import yfinance as yf
import requests

class QuantumAIModel:
    def __init__(self):
        self.lstm_model = None
        self.ensemble_model = None
        self.scaler = None
        
    def create_lstm_model(self, input_shape):
        """Crée un modèle LSTM avancé pour le trading"""
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            LSTM(100, return_sequences=True, dropout=0.2),
            LSTM(50, dropout=0.2),
            Dense(100, activation='relu'),
            Dropout(0.3),
            Dense(50, activation='relu'),
            Dropout(0.2),
            Dense(3, activation='softmax')  # SELL, HOLD, BUY
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def create_ensemble_model(self):
        """Crée un modèle ensemble avec plusieurs algorithmes"""
        from sklearn.ensemble import VotingClassifier
        
        estimators = [
            ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
            ('xgb', self.get_xgboost_classifier())
        ]
        
        ensemble = VotingClassifier(estimators=estimators, voting='soft')
        return ensemble
    
    def get_xgboost_classifier(self):
        """Retourne un classifieur XGBoost"""
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
        except:
            from sklearn.ensemble import ExtraTreesClassifier
            return ExtraTreesClassifier(n_estimators=100, random_state=42)
    
    def extract_advanced_features(self, data):
        """Extrait des features techniques avancées"""
        # Indicateurs de tendance
        data['sma_20'] = ta.trend.sma_indicator(data['close'], window=20)
        data['sma_50'] = ta.trend.sma_indicator(data['close'], window=50)
        data['ema_12'] = ta.trend.ema_indicator(data['close'], window=12)
        data['ema_26'] = ta.trend.ema_indicator(data['close'], window=26)
        
        # MACD
        macd = ta.trend.MACD(data['close'])
        data['macd'] = macd.macd()
        data['macd_signal'] = macd.macd_signal()
        data['macd_histogram'] = macd.macd_diff()
        
        # RSI
        data['rsi'] = ta.momentum.RSIIndicator(data['close']).rsi()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(data['close'])
        data['bb_upper'] = bollinger.bollinger_hband()
        data['bb_lower'] = bollinger.bollinger_lband()
        data['bb_middle'] = bollinger.bollinger_mavg()
        data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(data['high'], data['low'], data['close'])
        data['stoch_k'] = stoch.stoch()
        data['stoch_d'] = stoch.stoch_signal()
        
        # Volume indicators
        data['volume_sma'] = ta.volume.volume_sma(data['volume'], window=20)
        data['volume_ratio'] = data['volume'] / data['volume_sma']
        
        # ATR pour la volatilité
        data['atr'] = ta.volatility.AverageTrueRange(data['high'], data['low'], data['close']).average_true_range()
        
        # Price changes
        data['price_change_1h'] = data['close'].pct_change(1)
        data['price_change_4h'] = data['close'].pct_change(4)
        data['price_change_24h'] = data['close'].pct_change(24)
        
        # Volatility
        data['volatility'] = data['price_change_1h'].rolling(window=24).std()
        
        return data.dropna()