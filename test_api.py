import requests
import json

# URL de votre application Render
url = "https://votre-app.onrender.com/scalping-predict"

# Données de test
payload = {
    "symbol": "BTCUSD",
    "features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 
                 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
                 2.1, 2.2, 2.3, 2.4, 2.5]
}

headers = {
    "Content-Type": "application/json"
}

try:
    print("🚀 Envoi de la requête de test...")
    response = requests.post(url, json=payload, headers=headers, timeout=30)
    
    print(f"📊 Status Code: {response.status_code}")
    print(f"📦 Response: {response.text}")
    
    if response.status_code == 200:
        data = response.json()
        print("\n✅ SUCCÈS - Signal de trading reçu:")
        print(f"   Symbole: {data.get('symbol')}")
        print(f"   Action: {data.get('action')} ({'SELL' if data.get('action') == -1 else 'HOLD' if data.get('action') == 0 else 'BUY'})")
        print(f"   Confiance: {data.get('confidence'):.2%}")
        print(f"   SL: {data.get('scalping_sl')}")
        print(f"   TP: {data.get('scalping_tp')}")
        print(f"   Lots: {data.get('suggested_lots')}")
    else:
        print(f"❌ ERREUR: {response.status_code} - {response.text}")
        
except requests.exceptions.RequestException as e:
    print(f"❌ Erreur de connexion: {e}")
except Exception as e:
    print(f"❌ Erreur inattendue: {e}")