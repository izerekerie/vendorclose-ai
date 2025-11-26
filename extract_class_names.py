"""
Script to extract class names from data/train directory and save to JSON
This file should be included in the repository for deployment
"""

import json
from pathlib import Path

def extract_class_names(data_dir="data/train"):
    """Extract class names from data directory structure"""
    train_dir = Path(data_dir)
    
    if not train_dir.exists():
        print(f"Warning: {train_dir} does not exist")
        return None
    
    # Get all subdirectories (class folders)
    class_dirs = [d for d in train_dir.iterdir() if d.is_dir()]
    class_names = sorted([d.name for d in class_dirs])
    
    print(f"Found {len(class_names)} classes:")
    for i, name in enumerate(class_names, 1):
        print(f"  {i}. {name}")
    
    return class_names

def save_class_names(class_names, output_file="models/class_names.json"):
    """Save class names to JSON file"""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(class_names, f, indent=2)
    
    print(f"\n✅ Class names saved to {output_path}")
    return output_path

if __name__ == "__main__":
    print("Extracting class names from data/train directory...\n")
    class_names = extract_class_names()
    
    if class_names:
        save_class_names(class_names)
        print(f"\n📝 Total classes: {len(class_names)}")
    else:
        print("❌ Failed to extract class names")

