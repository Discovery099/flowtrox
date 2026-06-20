"""Quick test for optimization endpoints."""

import requests
import time

BASE_URL = "https://dev-factory-10.preview.emergentagent.com"

print("=" * 80)
print("Testing Walk-Forward Optimization Endpoints")
print("=" * 80)

# Start optimization
print("\n1. Starting optimization...")
response = requests.post(f"{BASE_URL}/api/optimize/start", json={"symbol": "ES"})
print(f"   Status: {response.status_code}")

if response.status_code == 200:
    data = response.json()
    job_id = data.get('job_id')
    print(f"   Job ID: {job_id}")
    
    # Poll for status
    print("\n2. Polling optimization status (up to 6 minutes)...")
    max_wait = 360
    poll_interval = 5
    elapsed = 0
    
    while elapsed < max_wait:
        status_response = requests.get(f"{BASE_URL}/api/optimize/status/{job_id}")
        if status_response.status_code == 200:
            status_data = status_response.json()
            status = status_data.get('status')
            pct = status_data.get('pct', 0)
            
            print(f"   [{elapsed}s] Status: {status}, Progress: {pct}%")
            
            if status == 'done':
                print("\n✅ Optimization completed!")
                result = status_data.get('result', {})
                print(f"   Best params: {result.get('best_params')}")
                print(f"   Walk-forward Sharpe: {result.get('walk_forward_sharpe')}")
                print(f"   Has sensitivity: {'sensitivity' in result}")
                break
            elif status == 'failed':
                print(f"\n❌ Optimization failed: {status_data.get('error')}")
                break
        
        time.sleep(poll_interval)
        elapsed += poll_interval
    
    if elapsed >= max_wait:
        print("\n⚠️  Optimization did not complete within 6 minutes")
else:
    print(f"   ❌ Failed to start optimization: {response.text}")

print("\n" + "=" * 80)
