"""
Test SearchEngine functionality
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engines.search_engine import SearchEngine


def test_encode_text():
    print("\n" + "="*80)
    print("Test 1: Encode Text")
    print("="*80)
    
    engine = SearchEngine()
    
    test_texts = [
        "可爱的小猫",
        "美丽的风景",
        "美味的食物",
        "快乐的家庭",
    ]
    
    for text in test_texts:
        result = engine.process({
            "action": "encode_text",
            "text": text,
        })
        print(f"\nText: {text}")
        print(f"Feature shape: {result['shape']}")
        print(f"Feature norm: {sum(x**2 for x in result['features'][0])**0.5:.4f}")


def test_search_by_text():
    print("\n" + "="*80)
    print("Test 2: Search by Text (requires images)")
    print("="*80)
    
    engine = SearchEngine()
    
    print("\nNote: This test requires actual image files.")
    print("Example usage:")
    print("""
result = engine.process({
    "action": "search_by_text",
    "text": "可爱的小猫",
    "image_paths": ["cat1.jpg", "cat2.jpg", "dog.jpg", "landscape.jpg"],
    "top_k": 3
})

# Expected output:
{
    "query": "可爱的小猫",
    "results": [
        {"image_path": "cat1.jpg", "similarity": 0.85},
        {"image_path": "cat2.jpg", "similarity": 0.82},
        {"image_path": "dog.jpg", "similarity": 0.45}
    ]
}
    """)


def test_compare_images():
    print("\n" + "="*80)
    print("Test 3: Compare Images (requires images)")
    print("="*80)
    
    engine = SearchEngine()
    
    print("\nNote: This test requires actual image files.")
    print("Example usage:")
    print("""
result = engine.process({
    "action": "compare_images",
    "image_path1": "photo1.jpg",
    "image_path2": "photo2.jpg"
})

# Expected output:
{
    "image_path1": "photo1.jpg",
    "image_path2": "photo2.jpg",
    "similarity": 0.85
}
    """)


def main():
    print("\n" + "="*80)
    print("SearchEngine Test Suite")
    print("="*80)
    
    try:
        test_encode_text()
        test_search_by_text()
        test_compare_images()
        
        print("\n" + "="*80)
        print("Test Summary")
        print("="*80)
        print("\n✓ Text encoding test passed")
        print("⚠ Image tests require actual image files")
        print("\nNext steps:")
        print("  1. Prepare test images")
        print("  2. Run image search and comparison tests")
        print("  3. Integrate into your application")
        print("="*80 + "\n")
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease export models first:")
        print("  python clip_model_manager.py --download --export --model ViT-B-16")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()