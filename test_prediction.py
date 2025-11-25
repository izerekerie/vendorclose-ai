"""
Test prediction endpoint with an actual image
Make sure API is running first!
"""

import requests
import sys
from pathlib import Path

API_URL = "http://localhost:8000"


def test_prediction(image_path):
    """Test prediction with a single image"""
    
    print("🧪 Testing Prediction Endpoint")
    print("=" * 50)
    
    # Check if image exists
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        print("\nPlease provide a valid image path.")
        print("Example: data/test/fresh/apple_1.jpg")
        return False
    
    print(f"📸 Testing with image: {image_path}")
    
    try:
        # Make prediction request
        with open(image_path, 'rb') as f:
            files = {'file': (Path(image_path).name, f, 'image/jpeg')}
            response = requests.post(
                f"{API_URL}/predict",
                files=files,
                timeout=10
            )
        
        if response.status_code == 200:
            result = response.json()
            
            print("\n✅ Prediction successful!")
            print("-" * 50)
            print(f"Class: {result['class_name'].upper()}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"Action: {result['action']}")
            print("\nProbabilities:")
            for class_name, prob in result['probabilities'].items():
                print(f"  {class_name}: {prob:.2%}")
            
            return True
        else:
            print(f"\n❌ Error: Status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("\n❌ Cannot connect to API!")
        print("Make sure API is running:")
        print("  python -m uvicorn api.main:app --host 0.0.0.0 --port 8000")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


if __name__ == "__main__":
    # Get image path from command line or use default
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Try to find a test image
        test_paths = [
            "data/test/fresh/apple_1.jpg",
            "data/test/fresh/apple_2.jpg",
            "data/test/rotten/apple_1.jpg",
            "data/train/fresh/apple_1.jpg",
        ]
        
        image_path = None
        for path in test_paths:
            if Path(path).exists():
                image_path = path
                break
        
        if not image_path:
            print("❌ No test image found!")
            print("\nUsage: python test_prediction.py <image_path>")
            print("Example: python test_prediction.py data/test/fresh/apple_1.jpg")
            sys.exit(1)
    
    test_prediction(image_path)





