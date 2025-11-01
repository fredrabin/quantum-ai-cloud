import numpy as np
import pandas as pd
from datetime import datetime

class AdvancedRiskManager:
    def __init__(self, initial_balance=10000):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.positions = {}
        self.trade_history = []
        self.max_drawdown = 0
        self.consecutive_losses = 0
        
    def calculate_position_size(self, symbol, confidence, stop_loss_pct, risk_per_trade=0.02):
        """Calcule la taille de position avec gestion du risque de portefeuille"""
        
        # Ajustement basé sur la confiance
        confidence_factor = 0.5 + (confidence * 0.5)
        
        # Réduction après des pertes consécutives
        loss_penalty = max(0.3, 1 - (self.consecutive_losses * 0.1))
        
        # Risque ajusté
        adjusted_risk = risk_per_trade * confidence_factor * loss_penalty
        
        # Calcul du montant à risquer
        risk_amount = self.current_balance * adjusted_risk
        
        # Taille de position
        position_size = risk_amount / stop_loss_pct
        
        # Limite selon le symbole
        symbol_limits = {
            "XAUUSD": 0.5,   # Max 0.5 lot
            "BTCUSD": 0.1,   # Max 0.1 BTC
            "ETHUSD": 0.2,   # Max 0.2 ETH
            "USDJPY": 1.0,   # Max 1.0 lot
            "EURUSD": 1.0,   # Max 1.0 lot
            "SP500": 0.3     # Max 0.3 lot
        }
        
        max_size = symbol_limits.get(symbol, 0.5)
        position_size = min(position_size, max_size)
        
        return round(position_size, 2), adjusted_risk
    
    def update_balance(self, pnl):
        """Met à jour le solde et surveille le drawdown"""
        self.current_balance += pnl
        
        # Calcul du drawdown
        peak_balance = max(self.initial_balance, max([trade['balance'] for trade in self.trade_history] + [self.current_balance]))
        drawdown = (peak_balance - self.current_balance) / peak_balance
        self.max_drawdown = max(self.max_drawdown, drawdown)
        
        # Gestion des pertes consécutives
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
            
        # Réduction agressive du risque si drawdown important
        if self.max_drawdown > 0.1:  # 10% de drawdown
            return 0.5  # Réduction de 50% du risque
        elif self.max_drawdown > 0.05:  # 5% de drawdown
            return 0.7  # Réduction de 30% du risque
            
        return 1.0  # Pas de réduction
    
    def should_enter_trade(self, symbol, action, confidence, market_volatility):
        """Décide si on doit entrer dans un trade"""
        
        # Évite les trades pendant une forte volatilité
        if market_volatility > 0.05 and confidence < 0.8:
            return False
            
        # Évite les trades si trop de pertes consécutives
        if self.consecutive_losses >= 3 and confidence < 0.75:
            return False
            
        # Vérifie qu'on n'a pas déjà une position sur ce symbole
        if symbol in self.positions:
            return False
            
        return True