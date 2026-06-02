"""
Chinese-CLIP ONNX Usage Examples

This script demonstrates how to use exported ONNX models for:
1. Image similarity detection
2. Text-to-image search
3. Image tagging

Prerequisites:
    pip install onnxruntime cn-clip pillow numpy
"""

import numpy as np
from PIL import Image
from pathlib import Path

try:
    import onnxruntime as ort
except ImportError:
    print("Error: onnxruntime not installed")
    print("Install with: pip install onnxruntime")
    exit(1)

try:
    import cn_clip.clip as clip
except ImportError:
    print("Error: cn_clip not installed")
    print("Install with: pip install cn-clip")
    exit(1)


class ClipONNXEngine:
    """Chinese-CLIP ONNX Inference Engine"""
    
    def __init__(self, model_name: str = "ViT-B-16"):
        """
        Initialize ONNX engine
        
        Args:
            model_name: Model name (e.g., "ViT-B-16")
        """
        self.model_name = model_name
        self.models_dir = Path(__file__).parent / "models"
        
        vision_onnx = self.models_dir / f"clip_vision_{model_name.replace('-', '_').replace('/', '_')}.onnx"
        text_onnx = self.models_dir / f"clip_text_{model_name.replace('-', '_').replace('/', '_')}.onnx"
        
        if not vision_onnx.exists():
            raise FileNotFoundError(
                f"Vision ONNX not found: {vision_onnx}\n"
                f"Please run: python clip_model_manager.py --export --model {model_name}"
            )
        
        print(f"Loading vision encoder: {vision_onnx}")
        self.vision_session = ort.InferenceSession(str(vision_onnx))
        
        if text_onnx.exists():
            print(f"Loading text encoder: {text_onnx}")
            self.text_session = ort.InferenceSession(str(text_onnx))
        else:
            print(f"Warning: Text encoder ONNX not found, text search will be limited")
            self.text_session = None
        
        self._preprocess = self._build_preprocess()
        print("✓ ONNX engine initialized")
    
    def _build_preprocess(self):
        """Build preprocessing function"""
        def preprocess(image_path: str) -> np.ndarray:
            """
            Preprocess image for ONNX inference
            
            Args:
                image_path: Path to image file
            
            Returns:
                Preprocessed image array [1, 3, 224, 224]
            """
            image = Image.open(image_path).convert("RGB")
            image = image.resize((224, 224), Image.BILINEAR)
            image_array = np.array(image).astype(np.float32) / 255.0
            
            mean = np.array([0.48145466, 0.4578275, 0.40821073])
            std = np.array([0.26862954, 0.26130258, 0.27577711])
            image_array = (image_array - mean) / std
            
            image_array = image_array.transpose(2, 0, 1)
            image_array = np.expand_dims(image_array, 0)
            
            return image_array
        
        return preprocess
    
    def encode_image(self, image_path: str) -> np.ndarray:
        """
        Extract image features
        
        Args:
            image_path: Path to image file
        
        Returns:
            Normalized image features [1, feature_dim]
        """
        image_array = self._preprocess(image_path)
        features = self.vision_session.run(None, {"pixel_values": image_array})[0]
        features = features / np.linalg.norm(features, axis=-1, keepdims=True)
        return features
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Extract text features
        
        Args:
            text: Input text
        
        Returns:
            Normalized text features [1, feature_dim]
        """
        if self.text_session is None:
            raise RuntimeError("Text encoder not available")
        
        tokens = clip.tokenize([text])
        features = self.text_session.run(None, {"input_ids": tokens.numpy()})[0]
        features = features / np.linalg.norm(features, axis=-1, keepdims=True)
        return features
    
    def compute_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """
        Compute cosine similarity between two feature vectors
        
        Args:
            features1: First feature vector
            features2: Second feature vector
        
        Returns:
            Similarity score [0, 1]
        """
        return float(np.dot(features1, features2.T)[0][0])


def example_image_similarity():
    """Example 1: Image Similarity Detection"""
    print("\n" + "="*80)
    print("Example 1: Image Similarity Detection")
    print("="*80)
    
    engine = ClipONNXEngine("ViT-B-16")
    
    print("\nNote: This example requires actual image files.")
    print("Replace the image paths with your own images.")
    
    img1_path = "photo1.jpg"
    img2_path = "photo2.jpg"
    
    print(f"\nTo compute similarity between two images:")
    print(f"```python")
    print(f"feat1 = engine.encode_image('{img1_path}')")
    print(f"feat2 = engine.encode_image('{img2_path}')")
    print(f"similarity = engine.compute_similarity(feat1, feat2)")
    print(f"print(f'Similarity: {{similarity:.4f}}')")
    print(f"```")
    
    print(f"\nExpected output:")
    print(f"Similarity: 0.8542  (high similarity)")
    print(f"Similarity: 0.3214  (low similarity)")


def example_text_search():
    """Example 2: Text-to-Image Search"""
    print("\n" + "="*80)
    print("Example 2: Text-to-Image Search")
    print("="*80)
    
    engine = ClipONNXEngine("ViT-B-16")
    
    print("\nNote: This example requires actual image files.")
    print("Replace the image paths with your own images.")
    
    query_text = "可爱的小猫"
    image_paths = ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg", "img5.jpg"]
    
    print(f"\nTo search images by text:")
    print(f"```python")
    print(f"query_text = '{query_text}'")
    print(f"image_paths = {image_paths}")
    print(f"")
    print(f"# Extract text features")
    print(f"text_features = engine.encode_text(query_text)")
    print(f"")
    print(f"# Extract all image features")
    print(f"image_features = [engine.encode_image(p) for p in image_paths]")
    print(f"")
    print(f"# Compute similarities")
    print(f"similarities = [engine.compute_similarity(text_features, f) for f in image_features]")
    print(f"")
    print(f"# Sort by similarity")
    print(f"results = sorted(zip(image_paths, similarities), key=lambda x: x[1], reverse=True)")
    print(f"")
    print(f"# Print results")
    print(f"for img_path, score in results:")
    print(f"    print(f'{{img_path}}: {{score:.4f}}')")
    print(f"```")
    
    print(f"\nExpected output:")
    print(f"img3.jpg: 0.8234")
    print(f"img1.jpg: 0.7123")
    print(f"img5.jpg: 0.6543")
    print(f"img2.jpg: 0.4321")
    print(f"img4.jpg: 0.3214")


def example_image_tagging():
    """Example 3: Image Tagging"""
    print("\n" + "="*80)
    print("Example 3: Image Tagging")
    print("="*80)
    
    engine = ClipONNXEngine("ViT-B-16")
    
    print("\nNote: This requires pre-computed tag embeddings.")
    print("You can generate them using export_clip_onnx_v2.py")
    
    tags = ["猫", "狗", "风景", "人物", "美食", "建筑", "车辆", "植物"]
    image_path = "photo.jpg"
    
    print(f"\nTo tag an image:")
    print(f"```python")
    print(f"# Pre-computed tag embeddings (example)")
    print(f"tag_embeddings = {{")
    print(f"    '猫': engine.encode_text('猫'),")
    print(f"    '狗': engine.encode_text('狗'),")
    print(f"    '风景': engine.encode_text('风景'),")
    print(f"    '人物': engine.encode_text('人物'),")
    print(f"    ...")
    print(f"}}")
    print(f"")
    print(f"# Extract image features")
    print(f"image_features = engine.encode_image('{image_path}')")
    print(f"")
    print(f"# Compute similarities with all tags")
    print(f"tag_scores = {{")
    print(f"    tag: engine.compute_similarity(image_features, emb)")
    print(f"    for tag, emb in tag_embeddings.items()")
    print(f"}}")
    print(f"")
    print(f"# Sort by score")
    print(f"sorted_tags = sorted(tag_scores.items(), key=lambda x: x[1], reverse=True)")
    print(f"")
    print(f"# Print top tags")
    print(f"for tag, score in sorted_tags[:5]:")
    print(f"    print(f'{{tag}}: {{score:.4f}}')")
    print(f"```")
    
    print(f"\nExpected output:")
    print(f"猫: 0.8234")
    print(f"人物: 0.6543")
    print(f"风景: 0.4321")
    print(f"狗: 0.3214")
    print(f"美食: 0.2123")


def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("Chinese-CLIP ONNX Usage Examples")
    print("="*80)
    print("\nThese examples demonstrate how to use exported ONNX models")
    print("for common computer vision tasks.")
    
    try:
        example_image_similarity()
        example_text_search()
        example_image_tagging()
        
        print("\n" + "="*80)
        print("Summary")
        print("="*80)
        print("\n✓ All examples demonstrated")
        print("\nKey Points:")
        print("  1. Use encode_image() for image feature extraction")
        print("  2. Use encode_text() for text feature extraction")
        print("  3. Use compute_similarity() to compare features")
        print("  4. All features are normalized for cosine similarity")
        print("\nNext Steps:")
        print("  1. Prepare your own images")
        print("  2. Run the examples with actual data")
        print("  3. Integrate into your application")
        print("="*80 + "\n")
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease export models first:")
        print("  python clip_model_manager.py --download --export --model ViT-B-16")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()
