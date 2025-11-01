Write-Host "=== TEST QUANTUM AI API ===" -ForegroundColor Green

# Test santé
Write-Host "`n1. Testing health endpoint..." -ForegroundColor Yellow
try {
    $health = Invoke-RestMethod -Uri "https://quantumaitrader.onrender.com/health" -Method Get
    Write-Host "✅ HEALTH: $($health.status)" -ForegroundColor Green
} catch {
    Write-Host "❌ HEALTH ERROR: $($_.Exception.Message)" -ForegroundColor Red
}

# Test BTCUSD
Write-Host "`n2. Testing BTCUSD prediction..." -ForegroundColor Yellow
$body1 = '{"symbol":"BTCUSD","features":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5]}'
try {
    $response1 = Invoke-RestMethod -Uri "https://quantumaitrader.onrender.com/scalping-predict" -Method Post -Body $body1 -ContentType "application/json"
    Write-Host "✅ BTCUSD Prediction SUCCESS!" -ForegroundColor Green
    Write-Host "   Action: $($response1.action), Confidence: $($response1.confidence)" -ForegroundColor White
} catch {
    Write-Host "❌ BTCUSD ERROR: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "`n=== TEST COMPLETED ===" -ForegroundColor Green
pause