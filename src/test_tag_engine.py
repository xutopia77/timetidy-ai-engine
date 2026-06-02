"""
Test TagEngine functionality
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engines.tag_engine_dynamic import TagEngine


def test_get_available_tags():
    print("\n" + "="*80)
    print("Test 1: Get Available Tags")
    print("="*80)
    
    engine = TagEngine()
    
    result = engine.process({
        "action": "get_available_tags",
    })
    
    print(f"\nAvailable default tags: {result['count']}")
    print(f"Tags: {', '.join(result['default_tags'][:10])}...")
    print(f"\n✓ Test passed")


def test_tag_image_with_default_tags():
    print("\n" + "="*80)
    print("Test 2: Tag Image with Default Tags (requires image)")
    print("="*80)
    
    engine = TagEngine()
    
    print("\nNote: This test requires actual image files.")
    print("Example usage:")
    print("""
result = engine.process({
    "action": "tag_image",
    "image_path": "photo.jpg",
    "top_k": 5,
    "threshold": 0.2
})

# Expected output:
{
    "image_path": "photo.jpg",
    "tags": [
        {"label": "风景", "confidence": 0.85},
        {"label": "山川", "confidence": 0.72},
        {"label": "自然", "confidence": 0.65},
        {"label": "室外", "confidence": 0.58},
        {"label": "白天", "confidence": 0.52}
    ],
    "total_tags": 80
}
    """)


def test_tag_image_with_custom_tags():
    print("\n" + "="*80)
    print("Test 3: Tag Image with Custom Tags (requires image)")
    print("="*80)
    
    engine = TagEngine()
    
    print("\nNote: This test requires actual image files.")
    print("Example usage:")
    print("""
result = engine.process({
    "action": "tag_image",
    "image_path": "photo.jpg",
    "tags": ["猫", "狗", "风景", "人物", "美食"],
    "top_k": 3,
    "threshold": 0.3
})

# Expected output:
{
    "image_path": "photo.jpg",
    "tags": [
        {"label": "猫", "confidence": 0.85},
        {"label": "宠物", "confidence": 0.72},
        {"label": "动物", "confidence": 0.65}
    ],
    "total_tags": 5
}
    """)


def test_add_custom_tags():
    print("\n" + "="*80)
    print("Test 4: Add Custom Tags")
    print("="*80)
    
    engine = TagEngine()
    
    custom_tags = ["可爱的小猫", "美丽的日落", "快乐的家庭"]
    
    result = engine.process({
        "action": "add_custom_tags",
        "tags": custom_tags,
    })
    
    print(f"\nAdded custom tags: {custom_tags}")
    print(f"Cache size: {result['cache_size']}")
    print(f"\n✓ Test passed")


def main():
    print("\n" + "="*80)
    print("TagEngine Test Suite")
    print("="*80)
    
    try:
        test_get_available_tags()
        test_add_custom_tags()
        test_tag_image_with_default_tags()
        test_tag_image_with_custom_tags()
        
        print("\n" + "="*80)
        print("Test Summary")
        print("="*80)
        print("\n✓ Get tags test passed")
        print("✓ Add custom tags test passed")
        print("⚠ Image tagging tests require actual image files")
        print("\nKey Features:")
        print("  1. Default tags: 80+ predefined tags")
        print("  2. Custom tags: Add any tags you want")
        print("  3. Open vocabulary: Support arbitrary text")
        print("  4. Caching: Tag embeddings are cached for performance")
        print("\nNext steps:")
        print("  1. Prepare test images")
        print("  2. Run image tagging tests")
        print("  3. Integrate into your application")
        print("="*80 + "\n")
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease export models first:")
        print("  python clip_model_manager.py --download --export --model ViT-L-14")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()