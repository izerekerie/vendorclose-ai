"""
Simple test script to verify API is working
"""

import requests
import time

API_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint"""
    print("Testing /health endpoint...")
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Response: {response.json()}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


def test_stats():
    """Test stats endpoint"""
    print("\nTesting /stats endpoint...")
    try:
        response = requests.get(f"{API_URL}/stats", timeout=5)
        if response.status_code == 200:
            print("✅ Stats endpoint working")
            stats = response.json()
            print(f"   Model loaded: {stats.get('model_loaded', False)}")
            print(f"   Total images: {stats.get('total_images', 0)}")
            return True
        else:
            print(f"❌ Stats endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Stats endpoint failed: {e}")
        return False


def test_retrain_status():
    """Test retrain status endpoint"""
    print("\nTesting /retrain/status endpoint...")
    try:
        response = requests.get(f"{API_URL}/retrain/status", timeout=5)
        if response.status_code == 200:
            print("✅ Retrain status endpoint working")
            status = response.json()
            print(f"   Status: {status.get('status', 'unknown')}")
            return True
        else:
            print(f"❌ Retrain status failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Retrain status failed: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Testing VendorClose AI API")
    print("=" * 50)
    
    # Wait a bit for API to be ready
    print("Waiting for API to be ready...")
    time.sleep(2)
    
    results = []
    results.append(test_health())
    results.append(test_stats())
    results.append(test_retrain_status())
    
    print("\n" + "=" * 50)
    if all(results):
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed")
        print("\nMake sure the API is running:")
        print("  python -m uvicorn api.main:app --host 0.0.0.0 --port 8000")

