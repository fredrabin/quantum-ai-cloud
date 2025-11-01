import pandas as pd
import yfinance as yf
import requests
import time
from datetime import datetime, timedelta
import numpy as np

class MarketDataFetcher:
    def __init__(self):
        self.symbols = {
            "XAUUSD": "GC=F",  # Gold
            "BTCUSD": "BTC-USD",  # Bitcoin
            "ETHUSD": "ETH-USD",  # Ethereum
            "USDJPY": "JPY=X",  # USD/JPY
            "EURUSD": "EURUSD=X",  # EUR/USD
            "SP500": "^GSPC"  # S&P 500
        }
        
    def fetch_realtime_data(self, symbol, period="5d", interval="5m"):
        """Récupère les données en temps réel depuis Yahoo Finance"""
        try:
            yf_symbol = self.symbols.get(symbol)
            if not yf_symbol:
                raise ValueError(f"Symbole {symbol} non supporté")
            
            ticker = yf.Ticker(yf_symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                # Fallback à des données simulées si Yahoo Finance échoue
                return self.create_sample_data()
            
            # Nettoyage des données
            data = data.reset_index()
            data['timestamp'] = pd.to_datetime(data['Date'])
            data = data.set_index('timestamp')
            data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
            data.columns = ['open', 'high', 'low', 'close', 'volume']
            
            return data
            
        except Exception as e:
            print(f"Erreur lors de la récupération des données pour {symbol}: {e}")
            return self.create_sample_data()
    
    def create_sample_data(self):
        """Crée des données d'exemple en cas d'échec de récupération"""
        dates = pd.date_range(end=datetime.now(), periods=100, freq='5min')
        data = pd.DataFrame(index=dates)
        data['open'] = np.random.normal(100, 10, len(dates)).cumsum()
        data['high'] = data['open'] + np.random.uniform(0, 5, len(dates))
        data['low'] = data['open'] - np.random.uniform(0, 5, len(dates))
        data['close'] = data['open'] + np.random.normal(0, 2, len(dates))
        data['volume'] = np.random.randint(1000, 10000, len(dates))
        
        return data
    
    def get_technical_features(self, data, symbol):
        """Calcule les indicateurs techniques pour l'IA"""
        from ai_model import QuantumAIModel
        ai_model = QuantumAIModel()
        
        # Ajoute tous les indicateurs techniques
        data_with_features = ai_model.extract_advanced_features(data.copy())
        
        # Sélectionne les 25 features les plus importantes
        feature_columns = [
            'rsi', 'macd', 'macd_signal', 'macd_histogram', 
            'sma_20', 'sma_50', 'ema_12', 'ema_26',
            'bb_upper', 'bb_lower', 'bb_middle', 'bb_width',
            'stoch_k', 'stoch_d', 'volume_ratio', 'atr',
            'price_change_1h', 'price_change_4h', 'price_change_24h',
            'volatility'
        ]
        
        # Complète avec d'autres colonnes si nécessaire
        available_features = [col for col in feature_columns if col in data_with_features.columns]
        features = data_with_features[available_features].iloc[-1].values
        
        # Normalise pour avoir exactement 25 features
        if len(features) > 25:
            features = features[:25]
        elif len(features) < 25:
            features = np.pad(features, (0, 25 - len(features)), 'constant')
        
        return features.tolist()