import flask
from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import joblib
import logging
import requests
import os
import pandas as pd
from datetime import datetime
import telegram
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
import threading
import time
import ta
from ta import add_all_ta_features
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
import warnings
warnings.filterwarnings('ignore')

# Configuration Flask
app = Flask(__name__)

# Configuration avancée pour le scalping
SYMBOLS = ["XAUUSD", "BTCUSD", "ETHUSD", "USDJPY", "EURUSD", "SP500"]

# Configuration détaillée par symbole
SYMBOL_CONFIG = {
    "XAUUSD": {
        "base_sl": 0.0015, 
        "base_tp": 0.0030, 
        "risk_per_trade": 0.02,  # 2% de risque par trade
        "lot_size": 100,  # 100 onces par lot standard
        "value_per_pip": 10,  # $10 par pip
        "max_lots": 0.5,
        "volatility_multiplier": 1.2
    },
    "BTCUSD": {
        "base_sl": 0.0020, 
        "base_tp": 0.0040, 
        "risk_per_trade": 0.015,  # 1.5% de risque
        "lot_size": 1,  # 1 BTC par lot
        "value_per_pip": 1,  # $1 par pip (simplifié)
        "max_lots": 0.1,
        "volatility_multiplier": 1.5
    },
    "ETHUSD": {
        "base_sl": 0.0025, 
        "base_tp": 0.0050, 
        "risk_per_trade": 0.015,
        "lot_size": 1,  # 1 ETH par lot
        "value_per_pip": 1,
        "max_lots": 0.2,
        "volatility_multiplier": 1.5
    },
    "USDJPY": {
        "base_sl": 0.0008, 
        "base_tp": 0.0016, 
        "risk_per_trade": 0.02,
        "lot_size": 100000,  # 100,000 unités par lot
        "value_per_pip": 9,  # $9 par pip
        "max_lots": 1.0,
        "volatility_multiplier": 1.1
    },
    "EURUSD": {
        "base_sl": 0.0006, 
        "base_tp": 0.0012, 
        "risk_per_trade": 0.02,
        "lot_size": 100000,
        "value_per_pip": 10,  # $10 par pip
        "max_lots": 1.0,
        "volatility_multiplier": 1.1
    },
    "SP500": {
        "base_sl": 0.0010, 
        "base_tp": 0.0020, 
        "risk_per_trade": 0.015,
        "lot_size": 1,  # 1 contrat
        "value_per_pip": 50,  # $50 par point
        "max_lots": 0.3,
        "volatility_multiplier": 1.3
    }
}

# Configuration Telegram
TELEGRAM_TOKEN = "8396377413:AAGtSWquXrolQR2LlqRdh3a75zd8Zt5UOfg"
CHAT_ID = None

# Variables globales
bot = telegram.Bot(token=TELEGRAM_TOKEN)
trade_history = []
active_positions = {}

# =============================================================================
# CLASSES AVANCÉES POUR LA GESTION DES TRADES
# =============================================================================

class PositionManager:
    """Gère les positions ouvertes avec trailing stop agressif"""
    
    def __init__(self):
        self.positions = {}
        self.break_even_triggered = {}
        
    def open_position(self, symbol, action, entry_price, sl, tp, lots, confidence):
        """Ouvre une nouvelle position"""
        position_id = f"{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        position = {
            'id': position_id,
            'symbol': symbol,
            'action': action,  # 1 pour BUY, -1 pour SELL
            'entry_price': entry_price,
            'current_price': entry_price,
            'initial_sl': sl,
            'initial_tp': tp,
            'current_sl': sl,
            'current_tp': tp,
            'lots': lots,
            'confidence': confidence,
            'open_time': datetime.utcnow(),
            'break_even_triggered': False,
            'trailing_start_price': entry_price + (tp - entry_price) * 0.25 if action == 1 else entry_price - (entry_price - tp) * 0.25,
            'trailing_active': False
        }
        
        self.positions[position_id] = position
        active_positions[position_id] = position
        
        return position_id
    
    def update_position_price(self, position_id, current_price):
        """Met à jour le prix actuel et gère le trailing stop"""
        if position_id not in self.positions:
            return None
            
        position = self.positions[position_id]
        position['current_price'] = current_price
        
        # Calcul du profit actuel en pips
        if position['action'] == 1:  # BUY
            profit_pips = (current_price - position['entry_price']) / position['entry_price']
            # Active le trailing stop si 25% du TP est atteint
            if current_price >= position['trailing_start_price'] and not position['trailing_active']:
                position['trailing_active'] = True
                position['current_sl'] = position['entry_price']  # Break even
                position['break_even_triggered'] = True
                
            # Trailing stop agressif
            if position['trailing_active']:
                new_sl = current_price - (position['initial_tp'] - position['entry_price']) * 0.1
                if new_sl > position['current_sl']:
                    position['current_sl'] = new_sl
                    
        else:  # SELL
            profit_pips = (position['entry_price'] - current_price) / position['entry_price']
            # Active le trailing stop si 25% du TP est atteint
            if current_price <= position['trailing_start_price'] and not position['trailing_active']:
                position['trailing_active'] = True
                position['current_sl'] = position['entry_price']  # Break even
                position['break_even_triggered'] = True
                
            # Trailing stop agressif
            if position['trailing_active']:
                new_sl = current_price + (position['entry_price'] - position['initial_tp']) * 0.1
                if new_sl < position['current_sl']:
                    position['current_sl'] = new_sl
        
        return position
    
    def check_position_exit(self, position_id, current_price):
        """Vérifie si la position doit être fermée"""
        if position_id not in self.positions:
            return None
            
        position = self.positions[position_id]
        
        if position['action'] == 1:  # BUY
            # Check SL
            if current_price <= position['current_sl']:
                return 'sl'
            # Check TP
            elif current_price >= position['current_tp']:
                return 'tp'
        else:  # SELL
            # Check SL
            if current_price >= position['current_sl']:
                return 'sl'
            # Check TP
            elif current_price <= position['current_tp']:
                return 'tp'
                
        return None
    
    def close_position(self, position_id, close_price, reason):
        """Ferme une position"""
        if position_id not in self.positions:
            return None
            
        position = self.positions[position_id]
        
        # Calcul du PnL
        if position['action'] == 1:  # BUY
            pnl_pips = (close_price - position['entry_price']) / position['entry_price']
        else:  # SELL
            pnl_pips = (position['entry_price'] - close_price) / position['entry_price']
        
        # Conversion en valeur monétaire (simplifié)
        config = SYMBOL_CONFIG[position['symbol']]
        pnl_value = pnl_pips * config['value_per_pip'] * position['lots'] * 10000
        
        position.update({
            'close_price': close_price,
            'close_time': datetime.utcnow(),
            'close_reason': reason,
            'pnl_pips': pnl_pips,
            'pnl_value': pnl_value,
            'duration': (datetime.utcnow() - position['open_time']).total_seconds() / 60  # en minutes
        })
        
        # Retire de les positions actives
        if position_id in active_positions:
            del active_positions[position_id]
            
        return position

class AdvancedRiskManager:
    def __init__(self, initial_balance=10000):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.positions = {}
        self.trade_history = []
        self.max_drawdown = 0
        self.consecutive_losses = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.position_manager = PositionManager()
        
    def calculate_dynamic_sl_tp(self, symbol, confidence, current_volatility):
        """Calcule SL et TP dynamiques basés sur la volatilité"""
        config = SYMBOL_CONFIG[symbol]
        
        # Ajustement basé sur la confiance
        confidence_factor = 0.7 + (confidence * 0.3)  # 0.7 à 1.0
        
        # Ajustement basé sur la volatilité
        volatility_factor = 1.0 + (current_volatility * config['volatility_multiplier'])
        
        # SL et TP dynamiques
        base_sl = config['base_sl']
        base_tp = config['base_tp']
        
        # SL plus serré avec haute confiance, plus large avec haute volatilité
        dynamic_sl = base_sl / confidence_factor * volatility_factor
        dynamic_tp = base_tp * confidence_factor * volatility_factor
        
        # Ratio TP/SL minimum de 1.5
        min_tp_sl_ratio = 1.5
        if dynamic_tp / dynamic_sl < min_tp_sl_ratio:
            dynamic_tp = dynamic_sl * min_tp_sl_ratio
        
        return round(dynamic_sl, 5), round(dynamic_tp, 5)
    
    def calculate_position_size(self, symbol, confidence, stop_loss_pips, current_price):
        """Calcule la taille de position précise basée sur le risque"""
        try:
            config = SYMBOL_CONFIG[symbol]
            
            # Ajustement basé sur la confiance
            confidence_factor = 0.5 + (confidence * 0.5)
            
            # Réduction après des pertes consécutives
            loss_penalty = max(0.3, 1 - (self.consecutive_losses * 0.15))
            
            # Risque ajusté par trade
            risk_per_trade = config['risk_per_trade'] * confidence_factor * loss_penalty
            
            # Montant à risquer en $
            risk_amount = self.current_balance * risk_per_trade
            
            # Calcul de la valeur du pip pour ce symbole
            pip_value = config['value_per_pip']
            
            # Calcul des lots basé sur le risque
            # risk_amount = lots * stop_loss_pips * pip_value
            position_size = risk_amount / (stop_loss_pips * pip_value * 10000)
            
            # Ajustement pour le prix actuel et la taille de lot
            if symbol in ["BTCUSD", "ETHUSD"]:
                # Pour les cryptos, ajustement basé sur le prix
                price_adjustment = current_price / 50000  # Référence BTC à 50k
                position_size = position_size / max(0.5, price_adjustment)
            
            # Application des limites
            position_size = max(0.01, min(position_size, config['max_lots']))
            
            # Arrondi selon le symbole
            if symbol in ["XAUUSD", "USDJPY", "EURUSD"]:
                position_size = round(position_size, 2)
            else:
                position_size = round(position_size, 3)
                
            return position_size, risk_per_trade
            
        except Exception as e:
            print(f"Erreur calcul position size: {e}")
            return 0.1, config['risk_per_trade']

    def update_balance(self, pnl):
        """Met à jour le solde et surveille le drawdown"""
        self.current_balance += pnl
        self.total_trades += 1
        
        if pnl > 0:
            self.winning_trades += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
        
        # Calcul du drawdown
        peak_balance = max(self.initial_balance, 
                          max([trade.get('balance', 0) for trade in self.trade_history] + [self.current_balance]))
        drawdown = (peak_balance - self.current_balance) / peak_balance if peak_balance > 0 else 0
        self.max_drawdown = max(self.max_drawdown, drawdown)
        
        # Ajoute au historique
        self.trade_history.append({
            'timestamp': datetime.utcnow(),
            'pnl': pnl,
            'balance': self.current_balance,
            'drawdown': drawdown
        })
        
        # Garde seulement les 100 derniers trades
        if len(self.trade_history) > 100:
            self.trade_history.pop(0)
            
        return self.get_risk_multiplier()
    
    def get_risk_multiplier(self):
        """Retourne le multiplicateur de risque basé sur la performance"""
        if self.max_drawdown > 0.1:  # 10% de drawdown
            return 0.5  # Réduction de 50% du risque
        elif self.max_drawdown > 0.05:  # 5% de drawdown
            return 0.7  # Réduction de 30% du risque
        elif self.consecutive_losses >= 3:
            return 0.6  # Réduction après 3 pertes consécutives
        elif self.current_balance < self.initial_balance * 0.9:
            return 0.8  # Réduction si en dessous du capital initial
            
        return 1.0

    def should_enter_trade(self, symbol, action, confidence, market_volatility):
        """Décide si on doit entrer dans un trade"""
        
        # Évite les trades pendant une forte volatilité
        if market_volatility > 0.08 and confidence < 0.8:
            return False, "Volatilité trop élevée"
            
        # Évite les trades si trop de pertes consécutives
        if self.consecutive_losses >= 3 and confidence < 0.75:
            return False, "Trop de pertes consécutives"
            
        # Vérifie qu'on n'a pas déjà une position sur ce symbole
        for position in active_positions.values():
            if position['symbol'] == symbol:
                return False, "Position déjà ouverte sur ce symbole"
            
        # Évite les trades si confiance trop basse
        if confidence < 0.65:
            return False, "Confiance trop basse"
            
        # Vérifie le drawdown
        if self.max_drawdown > 0.15:
            return False, "Drawdown trop important"
            
        return True, "OK"
    
    def get_performance_stats(self):
        """Retourne les statistiques de performance"""
        if self.total_trades == 0:
            return {
                'win_rate': 0,
                'total_trades': 0,
                'current_balance': self.current_balance,
                'max_drawdown': self.max_drawdown,
                'profit_total': 0,
                'consecutive_losses': self.consecutive_losses
            }
        
        win_rate = self.winning_trades / self.total_trades
        profit_total = self.current_balance - self.initial_balance
        
        return {
            'win_rate': win_rate,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'current_balance': round(self.current_balance, 2),
            'max_drawdown': round(self.max_drawdown, 4),
            'profit_total': round(profit_total, 2),
            'consecutive_losses': self.consecutive_losses,
            'risk_multiplier': self.get_risk_multiplier()
        }

# =============================================================================
# CLASSES AI ET DATA FETCHER
# =============================================================================

class QuantumAIModel:
    def __init__(self):
        self.lstm_model = None
        self.ensemble_model = None
        
    def extract_advanced_features(self, data):
        """Extrait des features techniques avancées"""
        try:
            df = data.copy()
            
            # Indicateurs de tendance
            df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
            df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
            df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
            df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
            
            # MACD
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_histogram'] = macd.macd_diff()
            
            # RSI
            df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
            
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(df['close'])
            df['bb_upper'] = bollinger.bollinger_hband()
            df['bb_lower'] = bollinger.bollinger_lband()
            df['bb_middle'] = bollinger.bollinger_mavg()
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            
            # Stochastic
            stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
            
            # Volume indicators
            if 'volume' in df.columns:
                df['volume_sma'] = ta.volume.volume_sma(df['volume'], window=20)
                df['volume_ratio'] = df['volume'] / df['volume_sma']
            else:
                df['volume_ratio'] = 1.0
            
            # ATR pour la volatilité
            df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
            
            # Price changes et volatilité
            df['price_change_1h'] = df['close'].pct_change(1)
            df['price_change_4h'] = df['close'].pct_change(4)
            df['volatility'] = df['price_change_1h'].rolling(window=24, min_periods=1).std()
            
            return df.fillna(method='bfill').fillna(method='ffill')
            
        except Exception as e:
            print(f"Erreur extraction features: {e}")
            return data

class MarketDataFetcher:
    def __init__(self):
        self.symbols = {
            "XAUUSD": "GC=F", "BTCUSD": "BTC-USD", "ETHUSD": "ETH-USD",
            "USDJPY": "JPY=X", "EURUSD": "EURUSD=X", "SP500": "^GSPC"
        }
        
    def fetch_realtime_data(self, symbol, period="2d", interval="15m"):
        """Récupère les données en temps réel"""
        try:
            yf_symbol = self.symbols.get(symbol)
            if not yf_symbol:
                return self.create_sample_data()
            
            ticker = yf.Ticker(yf_symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                return self.create_sample_data()
            
            data = data.reset_index()
            if 'Date' in data.columns:
                data['timestamp'] = pd.to_datetime(data['Date'])
            else:
                data['timestamp'] = data.index
                
            data = data.set_index('timestamp')
            
            # Renommage des colonnes
            column_mapping = {'Open': 'open', 'High': 'high', 'Low': 'low', 
                            'Close': 'close', 'Volume': 'volume'}
            
            for old_col, new_col in column_mapping.items():
                if old_col in data.columns:
                    data[new_col] = data[old_col]
            
            required_cols = ['open', 'high', 'low', 'close']
            for col in required_cols:
                if col not in data.columns:
                    data[col] = 100
                    
            if 'volume' not in data.columns:
                data['volume'] = 1000
                
            return data[['open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            print(f"Erreur récupération données {symbol}: {e}")
            return self.create_sample_data()
    
    def create_sample_data(self):
        """Crée des données d'exemple"""
        dates = pd.date_range(end=datetime.now(), periods=100, freq='15min')
        data = pd.DataFrame(index=dates)
        data['open'] = 100 + np.cumsum(np.random.normal(0, 0.5, len(dates)))
        data['high'] = data['open'] + np.random.uniform(0.1, 2.0, len(dates))
        data['low'] = data['open'] - np.random.uniform(0.1, 2.0, len(dates))
        data['close'] = (data['high'] + data['low']) / 2 + np.random.normal(0, 0.2, len(dates))
        data['volume'] = np.random.randint(1000, 10000, len(dates))
        return data
    
    def get_current_price(self, symbol):
        """Récupère le prix actuel pour un symbole"""
        try:
            data = self.fetch_realtime_data(symbol, period="1d", interval="5m")
            return float(data['close'].iloc[-1]) if len(data) > 0 else 100.0
        except:
            return 100.0
    
    def get_technical_features(self, data, symbol):
        """Calcule les indicateurs techniques"""
        try:
            ai_model = QuantumAIModel()
            data_with_features = ai_model.extract_advanced_features(data.copy())
            
            priority_features = [
                'rsi', 'macd', 'macd_signal', 'macd_histogram', 
                'sma_20', 'sma_50', 'ema_12', 'ema_26',
                'bb_upper', 'bb_lower', 'bb_middle', 'bb_width',
                'stoch_k', 'stoch_d', 'volume_ratio', 'atr',
                'price_change_1h', 'price_change_4h', 'volatility'
            ]
            
            available_features = []
            for feature in priority_features:
                if feature in data_with_features.columns:
                    value = data_with_features[feature].iloc[-1] if len(data_with_features) > 0 else 0
                    available_features.append(float(value) if pd.notna(value) else 0.0)
                else:
                    available_features.append(0.0)
            
            # Normalise à 25 features
            if len(available_features) > 25:
                features = available_features[:25]
            elif len(available_features) < 25:
                features = available_features + [0.0] * (25 - len(available_features))
            else:
                features = available_features
                
            return features
            
        except Exception as e:
            print(f"Erreur calcul features techniques: {e}")
            return [0.0] * 25

# =============================================================================
# INITIALISATION ET FONCTIONS PRINCIPALES
# =============================================================================

ai_model = QuantumAIModel()
data_fetcher = MarketDataFetcher()
risk_manager = AdvancedRiskManager(initial_balance=10000)

def real_ai_prediction(symbol, features):
    """Prédiction IA avancée avec analyse technique"""
    try:
        if len(features) >= 19:
            rsi, macd, macd_signal = features[0], features[1], features[2]
            stoch_k, stoch_d, bb_width = features[12], features[13], features[10]
            volatility = features[18]
        else:
            rsi, macd, macd_signal, stoch_k, stoch_d, bb_width, volatility = 50, 0, 0, 50, 50, 0, 0.02
        
        # Logique de trading multi-indicateurs
        buy_signals = sell_signals = 0
        
        # RSI
        if rsi < 30: buy_signals += 2
        elif rsi > 70: sell_signals += 2
        
        # MACD
        if macd > macd_signal and macd > 0: buy_signals += 1.5
        elif macd < macd_signal and macd < 0: sell_signals += 1.5
        
        # Stochastic
        if stoch_k < 20 and stoch_d < 20: buy_signals += 1
        elif stoch_k > 80 and stoch_d > 80: sell_signals += 1
        
        # Décision finale
        if buy_signals > sell_signals + 1:
            action, base_confidence = 1, min(0.95, buy_signals / 4.5)
        elif sell_signals > buy_signals + 1:
            action, base_confidence = -1, min(0.95, sell_signals / 4.5)
        else:
            action, base_confidence = 0, 0.6
        
        # Ajustement volatilité
        confidence = base_confidence * (0.8 if volatility > 0.05 else 1.0)
        confidence = max(0.5, min(0.95, confidence))
        
        # Probabilités
        if action == 1:
            probs = {"SELL": (1-confidence)*0.3, "HOLD": (1-confidence)*0.7, "BUY": confidence}
        elif action == -1:
            probs = {"SELL": confidence, "HOLD": (1-confidence)*0.7, "BUY": (1-confidence)*0.3}
        else:
            probs = {"SELL": (1-confidence)*0.4, "HOLD": confidence, "BUY": (1-confidence)*0.4}
        
        # Normalisation
        total = sum(probs.values())
        probabilities = {k: round(v/total, 3) for k, v in probs.items()}
        
        return action, confidence, probabilities
        
    except Exception as e:
        print(f"Erreur prédiction IA: {e}")
        return 0, 0.5, {"SELL": 0.333, "HOLD": 0.334, "BUY": 0.333}

# =============================================================================
# ROUTES FLASK
# =============================================================================

@app.route('/')
def home():
    return "🚀 Quantum AI Trading - Gestion Avancée des Lots & Risks"

@app.route('/health', methods=['GET'])
def health():
    stats = risk_manager.get_performance_stats()
    return jsonify({
        "status": "online",
        "timestamp": datetime.utcnow().isoformat(),
        "symbols": SYMBOLS,
        "performance": stats,
        "active_positions": len(active_positions),
        "version": "3.0 - Advanced Position Management"
    })

@app.route('/scalping-predict', methods=['POST'])
def scalping_predict():
    """Endpoint principal avec gestion avancée des positions"""
    try:
        data = request.get_json()
        symbol = data.get("symbol", "").upper()
        
        if symbol not in SYMBOLS:
            return jsonify({"error": f"Symbole non supporté. Valides: {SYMBOLS}"}), 400
        
        # Récupération données
        market_data = data_fetcher.fetch_realtime_data(symbol)
        features = data_fetcher.get_technical_features(market_data, symbol)
        current_price = data_fetcher.get_current_price(symbol)
        current_volatility = features[18] if len(features) > 18 else 0.02
        
        # Prédiction IA
        action, confidence, probabilities = real_ai_prediction(symbol, features)
        
        # Calcul SL/TP dynamiques
        sl, tp = risk_manager.calculate_dynamic_sl_tp(symbol, confidence, current_volatility)
        
        # Calcul taille position
        lots, risk_level = risk_manager.calculate_position_size(symbol, confidence, sl, current_price)
        
        # Vérification entrée trade
        should_trade, reason = risk_manager.should_enter_trade(symbol, action, confidence, current_volatility)
        
        # Préparation réponse
        response = {
            "symbol": symbol,
            "action": action,
            "action_text": "SELL" if action == -1 else "HOLD" if action == 0 else "BUY",
            "confidence": round(float(confidence), 4),
            "should_trade": should_trade,
            "trade_reason": reason,
            "current_price": round(current_price, 5),
            "scalping_sl": sl,
            "scalping_tp": tp,
            "suggested_lots": lots,
            "risk_level": round(float(risk_level), 4),
            "trailing_start": round(tp * 0.25, 5),
            "break_even_at": "25% du TP",
            "probabilities": probabilities,
            "timestamp": datetime.utcnow().isoformat(),
            "market_volatility": round(float(current_volatility), 5),
            "position_size_calculated": True
        }
        
        # Ouverture position si signal valide
        if should_trade and action != 0 and confidence > 0.65:
            position_id = risk_manager.position_manager.open_position(
                symbol, action, current_price, sl, tp, lots, confidence
            )
            response["position_id"] = position_id
            
            # Notification Telegram
            action_emoji = "🔴" if action == -1 else "🟢"
            message = (
                f"{action_emoji} *NOUVELLE POSITION* {action_emoji}\n"
                f"*Symbole:* {symbol}\n"
                f"*Action:* {'BUY' if action == 1 else 'SELL'}\n"
                f"*Prix entrée:* {current_price:.5f}\n"
                f"*SL:* {sl:.5f} | *TP:* {tp:.5f}\n"
                f"*Lots:* {lots} | *Risque:* {risk_level:.1%}\n"
                f"*Confiance:* {confidence:.1%}\n"
                f"*Trailing Start:* 25% TP → Break Even\n"
                f"*Volatilité:* {current_volatility:.3f}"
            )
            send_telegram_alert(message)
            
            trade_history.append({
                'symbol': symbol, 'action': action, 'confidence': confidence,
                'timestamp': datetime.utcnow(), 'lots': lots, 'price': current_price,
                'position_id': position_id
            })
        
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Erreur endpoint /scalping-predict: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/positions', methods=['GET'])
def get_positions():
    """Retourne les positions actives"""
    return jsonify({
        "active_positions": active_positions,
        "total_positions": len(active_positions),
        "timestamp": datetime.utcnow().isoformat()
    })

@app.route('/performance', methods=['GET'])
def get_performance():
    """Statistiques de performance détaillées"""
    stats = risk_manager.get_performance_stats()
    return jsonify({
        "performance": stats,
        "recent_trades": trade_history[-10:] if trade_history else [],
        "symbol_config": SYMBOL_CONFIG,
        "server_time": datetime.utcnow().isoformat()
    })

# =============================================================================
# FONCTIONS TELEGRAM
# =============================================================================

def send_telegram_alert(message):
    """Envoie une alerte Telegram"""
    try:
        if CHAT_ID:
            bot.send_message(chat_id=CHAT_ID, text=message, parse_mode='Markdown')
    except Exception as e:
        print(f"❌ Telegram error: {e}")

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Commande /start pour Telegram"""
    global CHAT_ID
    CHAT_ID = update.effective_chat.id
    stats = risk_manager.get_performance_stats()
    
    welcome_msg = (
        "🤖 *Quantum AI Trader - Version Avancée*\n\n"
        "*Fonctionnalités:* ✅\n"
        "• Gestion intelligente des lots\n" 
        "• SL/TP adaptatifs par symbole\n"
        "• Trailing stop agressif (BE à 25% TP)\n"
        "• Gestion capital avancée\n\n"
        f"*Performance:*\n"
        f"• Balance: ${stats['current_balance']:.2f}\n"
        f"• Win Rate: {stats['win_rate']:.1%}\n"
        f"• Risk Multiplier: {stats['risk_multiplier']:.1%}\n"
        f"• Positions actives: {len(active_positions)}"
    )
    
    await update.message.reply_text(welcome_msg, parse_mode='Markdown')

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Commande /stats pour Telegram"""
    stats = risk_manager.get_performance_stats()
    
    stats_msg = (
        "📈 *STATISTIQUES DÉTAILLÉES*\n"
        f"*Balance:* ${stats['current_balance']:.2f}\n"
        f"*Profit/Perte:* ${stats['profit_total']:.2f}\n"
        f"*Win Rate:* {stats['win_rate']:.1%}\n"
        f"*Total Trades:* {stats['total_trades']}\n"
        f"*Drawdown:* {stats['max_drawdown']:.1%}\n"
        f"*Pertes consécutives:* {stats['consecutive_losses']}\n"
        f"*Multiplicateur Risque:* {stats['risk_multiplier']:.0%}\n"
        f"*Positions actives:* {len(active_positions)}"
    )
    await update.message.reply_text(stats_msg, parse_mode='Markdown')

def run_telegram_bot():
    """Lance le bot Telegram"""
    try:
        application = Application.builder().token(TELEGRAM_TOKEN).build()
        application.add_handler(CommandHandler("start", start_command))
        application.add_handler(CommandHandler("stats", stats_command))
        application.run_polling()
    except Exception as e:
        print(f"❌ Erreur bot Telegram: {e}")

# =============================================================================
# DÉMARRAGE
# =============================================================================

if __name__ == '__main__':
    print("🚀 Démarrage Quantum AI - Gestion Avancée...")
    print("💰 Configuration des lots par symbole...")
    print("⚡ Trailing stop agressif activé...")
    print("📊 SL/TP adaptatifs à la volatilité...")
    
    # Test configuration
    for symbol, config in SYMBOL_CONFIG.items():
        print(f"   ✅ {symbol}: Lots max {config['max_lots']}, Risk {config['risk_per_trade']:.1%}")
    
    # Lancement bot Telegram
    try:
        telegram_thread = threading.Thread(target=run_telegram_bot, daemon=True)
        telegram_thread.start()
        print("✅ Bot Telegram démarré")
    except Exception as e:
        print(f"⚠️  Bot Telegram: {e}")
    
    print("🎯 Serveur prêt - Gestion avancée activée!")
    print("🌐 Health: /health")
    print("📈 Performance: /performance")
    print("📊 Positions: /positions")
    
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)