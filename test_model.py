"""
Standalone Model Testing Script
Tests the trained model without requiring retraining
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from src.prediction import FruitPredictor

def test_model():
    """Test the trained model on sample images"""
    
    print("\n" + "=" * 50)
    print("🧪 Model Testing - Standalone Script")
    print("=" * 50)
    
    # Paths
    model_path = Path('models/fruit_classifier.h5')
    data_dir = Path('data')
    test_dir = data_dir / 'test'
    
    # Check if model exists
    if not model_path.exists():
        print(f"❌ Error: Model not found at {model_path}")
        print("Please train the model first using the notebook or API.")
        return
    
    # Initialize predictor with correct image size (160x160 for trained model)
    print(f"\n📦 Loading model from: {model_path}")
    predictor = FruitPredictor(model_path=str(model_path), data_dir=str(data_dir), img_size=(160, 160))
    
    try:
        # Load model
        predictor.load_model(data_dir=str(data_dir))
        print(f"✅ Model loaded successfully!")
        print(f"   Classes detected: {len(predictor.class_names)}")
        print(f"   Class names: {predictor.class_names[:5]}..." if len(predictor.class_names) > 5 else f"   Class names: {predictor.class_names}")
        
        # Test on sample images
        if test_dir.exists():
            print(f"\n📸 Testing on sample images from: {test_dir}")
            
            # Get sample images from each class
            test_images = []
            for class_dir in test_dir.iterdir():
                if class_dir.is_dir():
                    images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
                    if images:
                        test_images.append(images[0])
            
            if test_images:
                # Limit to 10 images for display
                test_images = test_images[:10]
                print(f"\n   Found {len(test_images)} test images")
                
                # Create figure
                cols = min(5, len(test_images))
                rows = (len(test_images) + cols - 1) // cols
                fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
                
                if len(test_images) == 1:
                    axes = [axes]
                elif rows == 1:
                    axes = axes if isinstance(axes, np.ndarray) else [axes]
                else:
                    axes = axes.flatten()
                
                results_summary = []
                
                for i, img_path in enumerate(test_images):
                    try:
                        # Display image first (even if prediction fails)
                        ax = axes[i] if len(test_images) > 1 else axes[0]
                        img = plt.imread(img_path)
                        ax.imshow(img)
                        ax.axis('off')
                        
                        # Try prediction
                        result = predictor.predict_single(img_path)
                        results_summary.append(result)
                        
                        # Update title with prediction
                        ax.set_title(
                            f"{result['class']}\nConf: {result['confidence']:.1%}",
                            fontsize=9
                        )
                        
                        print(f"\n  ✅ {img_path.name}")
                        print(f"     Class: {result['class']}")
                        print(f"     Confidence: {result['confidence']:.2%}")
                        print(f"     Action: {result['action'][:50]}...")
                        
                    except Exception as e:
                        # Still show the image even if prediction fails
                        try:
                            ax.set_title(f"Error: {str(e)[:30]}", fontsize=8, color='red')
                        except:
                            pass
                        print(f"  ❌ Error processing {img_path.name}: {e}")
                        import traceback
                        traceback.print_exc()
                
                # Hide unused subplots
                for i in range(len(test_images), len(axes)):
                    axes[i].axis('off')
                
                plt.tight_layout()
                plt.savefig('logs/test_predictions.png', dpi=150, bbox_inches='tight')
                print("\n📊 Visualization saved to: logs/test_predictions.png")
                plt.show()
                
                # Summary statistics
                if results_summary:
                    print("\n📊 Test Summary:")
                    print(f"   Total images tested: {len(results_summary)}")
                    confidences = [r['confidence'] for r in results_summary]
                    print(f"   Average confidence: {np.mean(confidences):.2%}")
                    print(f"   Min confidence: {np.min(confidences):.2%}")
                    print(f"   Max confidence: {np.max(confidences):.2%}")
                
                print("\n✅ Testing completed successfully!")
            else:
                print("⚠️  No test images found in test directory")
        else:
            print(f"⚠️  Test directory not found: {test_dir}")
            print("   You can still test by providing image paths directly")
            
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    Path('logs').mkdir(exist_ok=True)
    
    # Run tests
    success = test_model()
    
    if success:
        print("\n" + "=" * 50)
        print("✅ All tests passed!")
        print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("❌ Some tests failed. Check the errors above.")
        print("=" * 50)
        sys.exit(1)

