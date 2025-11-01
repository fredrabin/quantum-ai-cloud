import flask
from flask import Flask, request, jsonify
import numpy as np
# TensorFlow removed for free tier compatibility
import joblib
import logging
import requests
import os
import pandas as pd
from datetime import datetime
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
        "risk_per_trade": 0.02,
        "lot_size": 100,
        "value_per_pip": 10,
        "max_lots": 0.5,
        "volatility_multiplier": 1.2,
        "mt5_symbol": "XAUUSD"
    },
    "BTCUSD": {
        "base_sl": 0.0020, 
        "base_tp": 0.0040, 
        "risk_per_trade": 0.015,
        "lot_size": 1,
        "value_per_pip": 1,
        "max_lots": 0.1,
        "volatility_multiplier": 1.5,
        "mt5_symbol": "BTCUSD"
    },
    "ETHUSD": {
        "base_sl": 0.0025, 
        "base_tp": 0.0050, 
        "risk_per_trade": 0.015,
        "lot_size": 1,
        "value_per_pip": 1,
        "max_lots": 0.2,
        "volatility_multiplier": 1.5,
        "mt5_symbol": "ETHUSD"
    },
    "USDJPY": {
        "base_sl": 0.0008, 
        "base_tp": 0.0016, 
        "risk_per_trade": 0.02,
        "lot_size": 100000,
        "value_per_pip": 9,
        "max_lots": 1.0,
        "volatility_multiplier": 1.1,
        "mt5_symbol": "USDJPY"
    },
    "EURUSD": {
        "base_sl": 0.0006, 
        "base_tp": 0.0012, 
        "risk_per_trade": 0.02,
        "lot_size": 100000,
        "value_per_pip": 10,
        "max_lots": 1.0,
        "volatility_multiplier": 1.1,
        "mt5_symbol": "EURUSD"
    },
    "SP500": {
        "base_sl": 0.0010, 
        "base_tp": 0.0020, 
        "risk_per_trade": 0.015,
        "lot_size": 1,
        "value_per_pip": 50,
        "max_lots": 0.3,
        "volatility_multiplier": 1.3,
        "mt5_symbol": "US500"
    }
}

# Configuration sécurisée - À MODIFIER AVEC TES VRAIES INFOS
BROKER_CONFIG = {
    "mt5_account": "11163964",      # Remplace par ton numéro
    "mt5_password": "PsdG0!uV",      # Remplace par ton mot de passe
    "mt5_server": "VantageInternational-Demo",
    "demo_mode": True,  # True pour démo, False pour compte réel
    "auto_trade": False  # True pour exécution auto, False pour signaux seulement
}

# Configuration Telegram
TELEGRAM_TOKEN = "8396377413:AAGtSWquXrolQR2LlqRdh3a75zd8Zt5UOfg"
CHAT_ID = None

# Variables globales
trade_history = []
active_positions = {}
mt5_connected = False

# =============================================================================
# CONNEXION META TRADER 5
# =============================================================================

class MT5Connector:
    def __init__(self):
        self.connected = False
        self.account_info = None
        
    def initialize_connection(self):
        """Initialise la connexion à MT5"""
        try:
            # Essaye d'importer MT5
            import MetaTrader5 as mt5
            
            if not mt5.initialize():
                error = mt5.last_error()
                print(f"❌ Échec initialisation MT5: {error}")
                return False
                
            # Utilise les variables d'environnement pour plus de sécurité
            account = os.getenv('MT5_ACCOUNT', BROKER_CONFIG["mt5_account"])
            password = os.getenv('MT5_PASSWORD', BROKER_CONFIG["mt5_password"]) 
            server = os.getenv('MT5_SERVER', BROKER_CONFIG["mt5_server"])
            
            authorized = mt5.login(
                login=int(account),
                password=password,
                server=server
            )
            
            if authorized:
                self.account_info = mt5.account_info()
                self.connected = True
                print(f"✅ Connecté à MT5 - Compte: {self.account_info.login}")
                print(f"💰 Balance: ${self.account_info.balance:.2f}")
                print(f"💼 Broker: {self.account_info.company}")
                print(f"⚡ Levrage: 1:{self.account_info.leverage}")
                return True
            else:
                error = mt5.last_error()
                print(f"❌ Échec connexion MT5: {error}")
                return False
                
        except ImportError:
            print("⚠️  MetaTrader5 non installé - Mode simulation activé")
            return False
        except Exception as e:
            print(f"❌ Erreur connexion MT5: {e}")
            return False
    
    def place_order(self, symbol, action, lots, sl, tp):
        """Place un ordre sur MT5"""
        try:
            import MetaTrader5 as mt5
            
            if not self.connected:
                print("❌ Non connecté à MT5")
                return False
                
            # Conversion symbole pour MT5
            mt5_symbol = SYMBOL_CONFIG.get(symbol, {}).get("mt5_symbol", symbol)
            
            # Vérifie si le symbole est disponible
            symbol_info = mt5.symbol_info(mt5_symbol)
            if symbol_info is None:
                print(f"❌ Symbole {mt5_symbol} non disponible")
                return False
                
            if not symbol_info.visible:
                if not mt5.symbol_select(mt5_symbol, True):
                    print(f"❌ Impossible d'activer le symbole {mt5_symbol}")
                    return False
            
            # Récupère les prix
            tick = mt5.symbol_info_tick(mt5_symbol)
            if tick is None:
                print(f"❌ Impossible de récupérer les prix pour {mt5_symbol}")
                return False
                
            # Détermine le prix et le type d'ordre
            if action == 1:  # BUY
                order_type = mt5.ORDER_TYPE_BUY
                price = tick.ask
                sl_price = price - (sl * price)
                tp_price = price + (tp * price)
            else:  # SELL
                order_type = mt5.ORDER_TYPE_SELL  
                price = tick.bid
                sl_price = price + (sl * price)
                tp_price = price - (tp * price)
            
            # Préparation de la requête
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": mt5_symbol,
                "volume": lots,
                "type": order_type,
                "price": price,
                "sl": sl_price,
                "tp": tp_price,
                "deviation": 20,
                "magic": 234000,
                "comment": "QuantumAI",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Envoi de l'ordre
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                print(f"❌ Erreur ordre {mt5_symbol}: {result.retcode}")
                return False
            else:
                print(f"✅ Ordre exécuté - {mt5_symbol} - Ticket: {result.order}")
                return {
                    "ticket": result.order,
                    "symbol": mt5_symbol,
                    "type": "BUY" if action == 1 else "SELL",
                    "lots": lots,
                    "price": price,
                    "sl": sl_price,
                    "tp": tp_price
                }
                
        except Exception as e:
            print(f"❌ Erreur execution ordre: {e}")
            return False
    
    def get_account_info(self):
        """Récupère les infos du compte"""
        if self.connected and self.account_info:
            return {
                "balance": self.account_info.balance,
                "equity": self.account_info.equity,
                "margin": self.account_info.margin,
                "free_margin": self.account_info.margin_free,
                "leverage": self.account_info.leverage,
                "currency": self.account_info.currency,
                "server": self.account_info.server
            }
        return None

# Initialisation MT5
mt5_connector = MT5Connector()

# =============================================================================
# CLASSES AVANCÉES POUR LA GESTION DES TRADES
# =============================================================================

class PositionManager:
    def __init__(self):
        self.positions = {}
        self.break_even_triggered = {}
        
    def open_position(self, symbol, action, entry_price, sl, tp, lots, confidence, mt5_ticket=None):
        """Ouvre une nouvelle position"""
        position_id = f"{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        position = {
            'id': position_id,
            'symbol': symbol,
            'action': action,
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
            'trailing_active': False,
            'mt5_ticket': mt5_ticket,
            'execution_mode': 'REAL' if mt5_ticket else 'DEMO'
        }
        
        self.positions[position_id] = position
        active_positions[position_id] = position
        
        return position_id

    # ... (le reste de la classe PositionManager reste identique)
    def update_position_price(self, position_id, current_price):
        if position_id not in self.positions:
            return None
        position = self.positions[position_id]
        position['current_price'] = current_price
        
        if position['action'] == 1:
            profit_pips = (current_price - position['entry_price']) / position['entry_price']
            if current_price >= position['trailing_start_price'] and not position['trailing_active']:
                position['trailing_active'] = True
                position['current_sl'] = position['entry_price']
                position['break_even_triggered'] = True
            if position['trailing_active']:
                new_sl = current_price - (position['initial_tp'] - position['entry_price']) * 0.1
                if new_sl > position['current_sl']:
                    position['current_sl'] = new_sl
        else:
            profit_pips = (position['entry_price'] - current_price) / position['entry_price']
            if current_price <= position['trailing_start_price'] and not position['trailing_active']:
                position['trailing_active'] = True
                position['current_sl'] = position['entry_price']
                position['break_even_triggered'] = True
            if position['trailing_active']:
                new_sl = current_price + (position['entry_price'] - position['initial_tp']) * 0.1
                if new_sl < position['current_sl']:
                    position['current_sl'] = new_sl
        return position
    
    def check_position_exit(self, position_id, current_price):
        if position_id not in self.positions:
            return None
        position = self.positions[position_id]
        if position['action'] == 1:
            if current_price <= position['current_sl']:
                return 'sl'
            elif current_price >= position['current_tp']:
                return 'tp'
        else:
            if current_price >= position['current_sl']:
                return 'sl'
            elif current_price <= position['current_tp']:
                return 'tp'
        return None
    
    def close_position(self, position_id, close_price, reason):
        if position_id not in self.positions:
            return None
        position = self.positions[position_id]
        if position['action'] == 1:
            pnl_pips = (close_price - position['entry_price']) / position['entry_price']
        else:
            pnl_pips = (position['entry_price'] - close_price) / position['entry_price']
        
        config = SYMBOL_CONFIG[position['symbol']]
        pnl_value = pnl_pips * config['value_per_pip'] * position['lots'] * 10000
        
        position.update({
            'close_price': close_price,
            'close_time': datetime.utcnow(),
            'close_reason': reason,
            'pnl_pips': pnl_pips,
            'pnl_value': pnl_value,
            'duration': (datetime.utcnow() - position['open_time']).total_seconds() / 60
        })
        
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
        config = SYMBOL_CONFIG[symbol]
        confidence_factor = 0.7 + (confidence * 0.3)
        volatility_factor = 1.0 + (current_volatility * config['volatility_multiplier'])
        base_sl = config['base_sl']
        base_tp = config['base_tp']
        dynamic_sl = base_sl / confidence_factor * volatility_factor
        dynamic_tp = base_tp * confidence_factor * volatility_factor
        min_tp_sl_ratio = 1.5
        if dynamic_tp / dynamic_sl < min_tp_sl_ratio:
            dynamic_tp = dynamic_sl * min_tp_sl_ratio
        return round(dynamic_sl, 5), round(dynamic_tp, 5)
    
    def calculate_position_size(self, symbol, confidence, stop_loss_pips, current_price):
        try:
            config = SYMBOL_CONFIG[symbol]
            confidence_factor = 0.5 + (confidence * 0.5)
            loss_penalty = max(0.3, 1 - (self.consecutive_losses * 0.15))
            risk_per_trade = config['risk_per_trade'] * confidence_factor * loss_penalty
            
            # Utilise le solde réel si connecté à MT5, sinon le solde simulé
            if mt5_connector.connected:
                account_info = mt5_connector.get_account_info()
                current_balance = account_info['balance'] if account_info else self.current_balance
            else:
                current_balance = self.current_balance
                
            risk_amount = current_balance * risk_per_trade
            pip_value = config['value_per_pip']
            position_size = risk_amount / (stop_loss_pips * pip_value * 10000)
            
            if symbol in ["BTCUSD", "ETHUSD"]:
                price_adjustment = current_price / 50000
                position_size = position_size / max(0.5, price_adjustment)
            
            position_size = max(0.01, min(position_size, config['max_lots']))
            
            if symbol in ["XAUUSD", "USDJPY", "EURUSD"]:
                position_size = round(position_size, 2)
            else:
                position_size = round(position_size, 3)
                
            return position_size, risk_per_trade
            
        except Exception as e:
            print(f"Erreur calcul position size: {e}")
            return 0.1, config['risk_per_trade']

    def update_balance(self, pnl):
        self.current_balance += pnl
        self.total_trades += 1
        if pnl > 0:
            self.winning_trades += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
        
        peak_balance = max(self.initial_balance, 
                          max([trade.get('balance', 0) for trade in self.trade_history] + [self.current_balance]))
        drawdown = (peak_balance - self.current_balance) / peak_balance if peak_balance > 0 else 0
        self.max_drawdown = max(self.max_drawdown, drawdown)
        
        self.trade_history.append({
            'timestamp': datetime.utcnow(),
            'pnl': pnl,
            'balance': self.current_balance,
            'drawdown': drawdown
        })
        
        if len(self.trade_history) > 100:
            self.trade_history.pop(0)
        return self.get_risk_multiplier()
    
    def get_risk_multiplier(self):
        if self.max_drawdown > 0.1:
            return 0.5
        elif self.max_drawdown > 0.05:
            return 0.7
        elif self.consecutive_losses >= 3:
            return 0.6
        elif self.current_balance < self.initial_balance * 0.9:
            return 0.8
        return 1.0

    def should_enter_trade(self, symbol, action, confidence, market_volatility):
        if market_volatility > 0.08 and confidence < 0.8:
            return False, "Volatilité trop élevée"
        if self.consecutive_losses >= 3 and confidence < 0.75:
            return False, "Trop de pertes consécutives"
        for position in active_positions.values():
            if position['symbol'] == symbol:
                return False, "Position déjà ouverte sur ce symbole"
        if confidence < 0.65:
            return False, "Confiance trop basse"
        if self.max_drawdown > 0.15:
            return False, "Drawdown trop important"
        return True, "OK"
    
    def get_performance_stats(self):
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

import ta
import yfinance as yf

class QuantumAIModel:
    def __init__(self):
        self.lstm_model = None
        self.ensemble_model = None
        
    def extract_advanced_features(self, data):
        try:
            df = data.copy()
            df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
            df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
            df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
            df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_histogram'] = macd.macd_diff()
            df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
            bollinger = ta.volatility.BollingerBands(df['close'])
            df['bb_upper'] = bollinger.bollinger_hband()
            df['bb_lower'] = bollinger.bollinger_lband()
            df['bb_middle'] = bollinger.bollinger_mavg()
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
            if 'volume' in df.columns:
                df['volume_sma'] = ta.volume.volume_sma(df['volume'], window=20)
                df['volume_ratio'] = df['volume'] / df['volume_sma']
            else:
                df['volume_ratio'] = 1.0
            df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
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
        dates = pd.date_range(end=datetime.now(), periods=100, freq='15min')
        data = pd.DataFrame(index=dates)
        data['open'] = 100 + np.cumsum(np.random.normal(0, 0.5, len(dates)))
        data['high'] = data['open'] + np.random.uniform(0.1, 2.0, len(dates))
        data['low'] = data['open'] - np.random.uniform(0.1, 2.0, len(dates))
        data['close'] = (data['high'] + data['low']) / 2 + np.random.normal(0, 0.2, len(dates))
        data['volume'] = np.random.randint(1000, 10000, len(dates))
        return data
    
    def get_current_price(self, symbol):
        try:
            data = self.fetch_realtime_data(symbol, period="1d", interval="5m")
            return float(data['close'].iloc[-1]) if len(data) > 0 else 100.0
        except:
            return 100.0
    
    def get_technical_features(self, data, symbol):
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
# INITIALISATION DES COMPOSANTS
# =============================================================================

ai_model = QuantumAIModel()
data_fetcher = MarketDataFetcher()
risk_manager = AdvancedRiskManager(initial_balance=10000)

def real_ai_prediction(symbol, features):
    try:
        if len(features) >= 19:
            rsi, macd, macd_signal = features[0], features[1], features[2]
            stoch_k, stoch_d, bb_width = features[12], features[13], features[10]
            volatility = features[18]
        else:
            rsi, macd, macd_signal, stoch_k, stoch_d, bb_width, volatility = 50, 0, 0, 50, 50, 0, 0.02
        
        buy_signals = sell_signals = 0
        
        if rsi < 30: buy_signals += 2
        elif rsi > 70: sell_signals += 2
        if macd > macd_signal and macd > 0: buy_signals += 1.5
        elif macd < macd_signal and macd < 0: sell_signals += 1.5
        if stoch_k < 20 and stoch_d < 20: buy_signals += 1
        elif stoch_k > 80 and stoch_d > 80: sell_signals += 1
        
        if buy_signals > sell_signals + 1:
            action, base_confidence = 1, min(0.95, buy_signals / 4.5)
        elif sell_signals > buy_signals + 1:
            action, base_confidence = -1, min(0.95, sell_signals / 4.5)
        else:
            action, base_confidence = 0, 0.6
        
        confidence = base_confidence * (0.8 if volatility > 0.05 else 1.0)
        confidence = max(0.5, min(0.95, confidence))
        
        if action == 1:
            probs = {"SELL": (1-confidence)*0.3, "HOLD": (1-confidence)*0.7, "BUY": confidence}
        elif action == -1:
            probs = {"SELL": confidence, "HOLD": (1-confidence)*0.7, "BUY": (1-confidence)*0.3}
        else:
            probs = {"SELL": (1-confidence)*0.4, "HOLD": confidence, "BUY": (1-confidence)*0.4}
        
        total = sum(probs.values())
        probabilities = {k: round(v/total, 3) for k, v in probs.items()}
        
        return action, confidence, probabilities
        
    except Exception as e:
        print(f"Erreur prédiction IA: {e}")
        return 0, 0.5, {"SELL": 0.333, "HOLD": 0.334, "BUY": 0.333}

# =============================================================================
# TELEGRAM BOT SIMPLIFIÉ
# =============================================================================

def send_telegram_alert(message):
    try:
        if CHAT_ID:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            payload = {
                'chat_id': CHAT_ID,
                'text': message,
                'parse_mode': 'Markdown'
            }
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                print(f"📱 Message Telegram envoyé à {CHAT_ID}")
            else:
                print(f"❌ Erreur envoi Telegram: {response.status_code}")
    except Exception as e:
        print(f"❌ Erreur Telegram: {e}")

@app.route('/webhook/telegram', methods=['POST'])
def telegram_webhook():
    try:
        data = request.get_json()
        message = data.get('message', {})
        text = message.get('text', '').strip()
        chat_id = message.get('chat', {}).get('id')
        
        global CHAT_ID
        if chat_id:
            CHAT_ID = chat_id
        
        if text == '/start' or text == '/status':
            stats = risk_manager.get_performance_stats()
            mt5_info = mt5_connector.get_account_info()
            
            if mt5_info and mt5_connector.connected:
                balance_info = f"💰 *Balance MT5:* ${mt5_info['balance']:.2f}\n⚡ *Free Margin:* ${mt5_info['free_margin']:.2f}\n🎯 *Broker:* {mt5_info['server']}"
                mode_info = "🟢 *MODE RÉEL* - Connecté à MT5"
            else:
                balance_info = f"💰 *Balance Démo:* ${stats['current_balance']:.2f}"
                mode_info = "🟡 *MODE DÉMO* - Simulation"
                
            welcome_msg = (
                f"🤖 *Quantum AI Trader*\n\n"
                f"{mode_info}\n"
                f"{balance_info}\n"
                f"📈 *Win Rate:* {stats['win_rate']:.1%}\n"
                f"🎯 *Trades:* {stats['total_trades']}\n\n"
                f"*Commandes:*\n"
                f"/start - Statut\n"
                f"/stats - Statistiques\n"
                f"/mt5 - Info MT5\n"
                f"/trades - Derniers trades"
            )
            send_telegram_alert(welcome_msg)
            
        elif text == '/stats':
            stats = risk_manager.get_performance_stats()
            mt5_info = mt5_connector.get_account_info()
            
            stats_msg = (
                "📈 *STATISTIQUES DÉTAILLÉES*\n"
                f"*Balance:* ${stats['current_balance']:.2f}\n"
                f"*Profit/Perte:* ${stats['profit_total']:.2f}\n"
                f"*Win Rate:* {stats['win_rate']:.1%}\n"
                f"*Total Trades:* {stats['total_trades']}\n"
                f"*Drawdown:* {stats['max_drawdown']:.1%}\n"
                f"*Pertes consécutives:* {stats['consecutive_losses']}"
            )
            
            if mt5_info:
                stats_msg += f"\n\n*MT5 RÉEL:*\n*Equity:* ${mt5_info['equity']:.2f}\n*Margin:* ${mt5_info['margin']:.2f}"
                
            send_telegram_alert(stats_msg)
            
        elif text == '/mt5':
            if mt5_connector.connected:
                mt5_info = mt5_connector.get_account_info()
                if mt5_info:
                    mt5_msg = (
                        "💼 *META TRADER 5*\n"
                        f"*Compte:* {BROKER_CONFIG['mt5_account']}\n"
                        f"*Serveur:* {mt5_info['server']}\n"
                        f"*Balance:* ${mt5_info['balance']:.2f}\n"
                        f"*Equity:* ${mt5_info['equity']:.2f}\n"
                        f"*Free Margin:* ${mt5_info['free_margin']:.2f}\n"
                        f"*Levrage:* 1:{mt5_info['leverage']}\n"
                        f"*Devise:* {mt5_info['currency']}\n"
                        f"*Statut:* 🟢 CONNECTÉ"
                    )
                else:
                    mt5_msg = "❌ Impossible de récupérer les infos MT5"
            else:
                mt5_msg = "❌ Non connecté à MT5\n💡 Mode démo actif"
            send_telegram_alert(mt5_msg)
            
        elif text == '/trades':
            recent_trades = trade_history[-5:] if trade_history else []
            if recent_trades:
                trades_msg = "📊 *5 DERNIERS TRADES*\n"
                for trade in reversed(recent_trades):
                    pnl_color = "🟢" if trade['pnl'] > 0 else "🔴"
                    trades_msg += f"{pnl_color} ${trade['pnl']:.2f} | Balance: ${trade['balance']:.2f}\n"
            else:
                trades_msg = "📊 Aucun trade récent"
            send_telegram_alert(trades_msg)
        
        return jsonify({"status": "ok"})
        
    except Exception as e:
        print(f"❌ Erreur webhook Telegram: {e}")
        return jsonify({"status": "error", "message": str(e)})

# =============================================================================
# ROUTES FLASK PRINCIPALES
# =============================================================================

@app.route('/')
def home():
    return "🚀 Quantum AI Trading - Connecté à MT5"

@app.route('/health', methods=['GET'])
def health():
    stats = risk_manager.get_performance_stats()
    mt5_info = mt5_connector.get_account_info()
    
    health_data = {
        "status": "online",
        "timestamp": datetime.utcnow().isoformat(),
        "symbols": SYMBOLS,
        "performance": stats,
        "active_positions": len(active_positions),
        "version": "4.0 - MT5 Integrated",
        "mt5_connected": mt5_connector.connected,
        "demo_mode": BROKER_CONFIG["demo_mode"]
    }
    
    if mt5_info:
        health_data["mt5_info"] = mt5_info
        
    return jsonify(health_data)

@app.route('/scalping-predict', methods=['POST'])
def scalping_predict():
    try:
        data = request.get_json()
        symbol = data.get("symbol", "").upper()
        
        if symbol not in SYMBOLS:
            return jsonify({"error": f"Symbole non supporté. Valides: {SYMBOLS}"}), 400
        
        market_data = data_fetcher.fetch_realtime_data(symbol)
        features = data_fetcher.get_technical_features(market_data, symbol)
        current_price = data_fetcher.get_current_price(symbol)
        current_volatility = features[18] if len(features) > 18 else 0.02
        
        action, confidence, probabilities = real_ai_prediction(symbol, features)
        sl, tp = risk_manager.calculate_dynamic_sl_tp(symbol, confidence, current_volatility)
        lots, risk_level = risk_manager.calculate_position_size(symbol, confidence, sl, current_price)
        should_trade, reason = risk_manager.should_enter_trade(symbol, action, confidence, current_volatility)
        
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
            "probabilities": probabilities,
            "timestamp": datetime.utcnow().isoformat(),
            "execution_mode": "REAL" if mt5_connector.connected and not BROKER_CONFIG["demo_mode"] else "DEMO"
        }
        
        # EXÉCUTION DU TRADE
        if should_trade and action != 0 and confidence > 0.65:
            mt5_ticket = None
            
            # Exécution réelle si connecté et mode auto
            if mt5_connector.connected and BROKER_CONFIG["auto_trade"] and not BROKER_CONFIG["demo_mode"]:
                mt5_result = mt5_connector.place_order(symbol, action, lots, sl, tp)
                if mt5_result:
                    mt5_ticket = mt5_result["ticket"]
            
            # Ouverture position (réelle ou simulée)
            position_id = risk_manager.position_manager.open_position(
                symbol, action, current_price, sl, tp, lots, confidence, mt5_ticket
            )
            response["position_id"] = position_id
            if mt5_ticket:
                response["mt5_ticket"] = mt5_ticket
            
            # Notification Telegram
            action_emoji = "🔴" if action == -1 else "🟢"
            mode_emoji = "💼" if mt5_ticket else "🎮"
            message = (
                f"{action_emoji} *SIGNAL TRADING* {mode_emoji}\n"
                f"*Symbole:* {symbol}\n"
                f"*Action:* {'BUY' if action == 1 else 'SELL'}\n"
                f"*Prix:* {current_price:.5f}\n"
                f"*SL:* {sl:.5f} | *TP:* {tp:.5f}\n"
                f"*Lots:* {lots} | *Risque:* {risk_level:.1%}\n"
                f"*Confiance:* {confidence:.1%}\n"
                f"*Mode:* {'RÉEL' if mt5_ticket else 'DÉMO'}"
            )
            if mt5_ticket:
                message += f"\n*Ticket MT5:* {mt5_ticket}"
                
            send_telegram_alert(message)
        
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Erreur endpoint /scalping-predict: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/positions', methods=['GET'])
def get_positions():
    return jsonify({
        "active_positions": active_positions,
        "total_positions": len(active_positions),
        "timestamp": datetime.utcnow().isoformat()
    })

@app.route('/performance', methods=['GET'])
def get_performance():
    stats = risk_manager.get_performance_stats()
    mt5_info = mt5_connector.get_account_info()
    
    performance_data = {
        "performance": stats,
        "recent_trades": trade_history[-10:] if trade_history else [],
        "symbol_config": SYMBOL_CONFIG,
        "server_time": datetime.utcnow().isoformat(),
        "mt5_connected": mt5_connector.connected
    }
    
    if mt5_info:
        performance_data["mt5_account"] = mt5_info
        
    return jsonify(performance_data)

# =============================================================================
# DÉMARRAGE DE L'APPLICATION
# =============================================================================

if __name__ == '__main__':
    print("🚀 Démarrage Quantum AI - Intégration MT5...")
    print("💰 Configuration des lots par symbole...")
    print("⚡ Trailing stop agressif activé...")
    
    # Test configuration symboles
    for symbol, config in SYMBOL_CONFIG.items():
        print(f"   ✅ {symbol}: Lots max {config['max_lots']}, Risk {config['risk_per_trade']:.1%}")
    
    # Connexion MT5
    print("\n🔗 Connexion à MetaTrader 5...")
    if mt5_connector.initialize_connection():
        mt5_connected = True
        print("✅ Connecté à MT5 avec succès!")
    else:
        print("💡 Mode démo activé - Pas de connexion MT5")
    
    # Configuration Telegram
    print("🤖 Configuration Telegram...")
    print("✅ Webhook Telegram configuré sur /webhook/telegram")
    
    print("\n🎯 Serveur prêt - Intégration MT5 complète!")
    print("🌐 Health: /health")
    print("📈 Performance: /performance") 
    print("📊 Positions: /positions")
    print(f"💼 MT5: {'🟢 CONNECTÉ' if mt5_connected else '🔴 DÉCONNECTÉ'}")
    print(f"🎮 Mode: {'DÉMO' if BROKER_CONFIG['demo_mode'] else 'RÉEL'}")
    
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
