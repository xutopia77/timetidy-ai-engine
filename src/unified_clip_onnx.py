"""
Unified Chinese-CLIP ONNX Interface

This module provides a unified interface for Chinese-CLIP ONNX models.
Models are stored separately for efficiency, but loaded together for convenience.

Benefits:
- Separate storage: Load only what you need (save memory)
- Unified interface: Easy to use (single class)
- Flexible: Choose between separate or combined loading
"""

import numpy as np
from pathlib import Path
from typing import Optional, Union, List, Tuple
import onnxruntime as ort

try:
    import cn_clip.clip as clip
except ImportError:
    clip = None


class UnifiedClipONNX:
    """
    Unified interface for Chinese-CLIP ONNX models
    
    Models are stored separately but can be loaded together or individually.
    """
    
    def __init__(
        self,
        model_name: str = "ViT-B-16",
        models_dir: Optional[Path] = None,
        load_vision: bool = True,
        load_text: bool = True
    ):
        """
        Initialize unified CLIP interface
        
        Args:
            model_name: Model name (e.g., "ViT-B-16")
            models_dir: Directory containing ONNX models
            load_vision: Whether to load vision encoder
            load_text: Whether to load text encoder
        """
        self.model_name = model_name
        self.models_dir = models_dir or Path(__file__).parent / "models"
        
        self._vision_session: Optional[ort.InferenceSession] = None
        self._text_session: Optional[ort.InferenceSession] = None
        
        vision_onnx = self.models_dir / f"clip_vision_{model_name.replace('-', '_').replace('/', '_')}.onnx"
        text_onnx = self.models_dir / f"clip_text_{model_name.replace('-', '_').replace('/', '_')}.onnx"
        
        if load_vision:
            if not vision_onnx.exists():
                raise FileNotFoundError(
                    f"Vision encoder not found: {vision_onnx}\n"
                    f"Run: python clip_model_manager.py --export --model {model_name}"
                )
            print(f"Loading vision encoder: {vision_onnx.name}")
            self._vision_session = ort.InferenceSession(str(vision_onnx))
        
        if load_text:
            if not text_onnx.exists():
                print(f"Warning: Text encoder not found: {text_onnx}")
                print("Text search will be unavailable")
            else:
                print(f"Loading text encoder: {text_onnx.name}")
                self._text_session = ort.InferenceSession(str(text_onnx))
        
        print(f"✓ Unified CLIP initialized (vision={load_vision}, text={load_text})")
    
    @property
    def has_vision(self) -> bool:
        """Check if vision encoder is loaded"""
        return self._vision_session is not None
    
    @property
    def has_text(self) -> bool:
        """Check if text encoder is loaded"""
        return self._text_session is not None
    
    def encode_image(self, image: Union[str, np.ndarray]) -> np.ndarray:
        """
        Encode image to feature vector
        
        Args:
            image: Image path or preprocessed array
        
        Returns:
            Normalized feature vector [1, dim]
        """
        if not self.has_vision:
            raise RuntimeError("Vision encoder not loaded")
        
        if isinstance(image, str):
            image = self._preprocess_image(image)
        
        features = self._vision_session.run(None, {"pixel_values": image})[0]
        features = features / np.linalg.norm(features, axis=-1, keepdims=True)
        return features
    
    def encode_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Encode text to feature vector
        
        Args:
            text: Text string or list of strings
        
        Returns:
            Normalized feature vector [batch, dim]
        """
        if not self.has_text:
            raise RuntimeError("Text encoder not loaded")
        
        if clip is None:
            raise ImportError("cn_clip not installed. Install with: pip install cn-clip")
        
        if isinstance(text, str):
            text = [text]
        
        tokens = clip.tokenize(text)
        features = self._text_session.run(None, {"input_ids": tokens.numpy()})[0]
        features = features / np.linalg.norm(features, axis=-1, keepdims=True)
        return features
    
    def compute_similarity(
        self,
        features1: np.ndarray,
        features2: np.ndarray
    ) -> Union[float, np.ndarray]:
        """
        Compute cosine similarity
        
        Args:
            features1: First feature vector(s)
            features2: Second feature vector(s)
        
        Returns:
            Similarity score(s)
        """
        sim = np.dot(features1, features2.T)
        if sim.size == 1:
            return float(sim[0, 0])
        return sim
    
    def _preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess image for inference"""
        from PIL import Image
        
        image = Image.open(image_path).convert("RGB")
        image = image.resize((224, 224), Image.BILINEAR)
        image_array = np.array(image).astype(np.float32) / 255.0
        
        mean = np.array([0.48145466, 0.4578275, 0.40821073])
        std = np.array([0.26862954, 0.26130258, 0.27577711])
        image_array = (image_array - mean) / std
        
        image_array = image_array.transpose(2, 0, 1)
        image_array = np.expand_dims(image_array, 0)
        
        return image_array
    
    def search_images(
        self,
        query: str,
        image_paths: List[str],
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Search images by text query
        
        Args:
            query: Text query
            image_paths: List of image paths
            top_k: Number of results
        
        Returns:
            List of (image_path, score) tuples
        """
        text_features = self.encode_text(query)
        
        image_features_list = []
        for img_path in image_paths:
            img_feat = self.encode_image(img_path)
            image_features_list.append(img_feat)
        image_features = np.vstack(image_features_list)
        
        similarities = self.compute_similarity(text_features, image_features)[0]
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        results = [(image_paths[i], float(similarities[i])) for i in top_indices]
        
        return results
    
    def tag_image(
        self,
        image_path: str,
        tags: List[str],
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Tag image with given tags
        
        Args:
            image_path: Image path
            tags: List of tags
            top_k: Number of tags to return
        
        Returns:
            List of (tag, score) tuples
        """
        image_features = self.encode_image(image_path)
        tag_features = self.encode_text(tags)
        
        similarities = self.compute_similarity(image_features, tag_features)[0]
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        results = [(tags[i], float(similarities[i])) for i in top_indices]
        
        return results


def create_clip_model(
    model_name: str = "ViT-B-16",
    mode: str = "both"
) -> UnifiedClipONNX:
    """
    Factory function to create CLIP model
    
    Args:
        model_name: Model name
        mode: "vision", "text", or "both"
    
    Returns:
        UnifiedClipONNX instance
    
    Examples:
        # For image similarity (vision only)
        model = create_clip_model("ViT-B-16", "vision")
        
        # For text search (both)
        model = create_clip_model("ViT-B-16", "both")
        
        # For text features only
        model = create_clip_model("ViT-B-16", "text")
    """
    load_vision = mode in ["vision", "both"]
    load_text = mode in ["text", "both"]
    
    return UnifiedClipONNX(
        model_name=model_name,
        load_vision=load_vision,
        load_text=load_text
    )


if __name__ == "__main__":
    print("\n" + "="*80)
    print("Unified CLIP ONNX Interface")
    print("="*80)
    
    print("\nBenefits of separate storage + unified interface:")
    print("  ✅ Memory efficient: Load only what you need")
    print("  ✅ Easy to use: Single class for all operations")
    print("  ✅ Flexible: Choose loading mode")
    
    print("\nUsage examples:")
    print("```python")
    print("# Example 1: Image similarity (vision only)")
    print("model = create_clip_model('ViT-B-16', 'vision')")
    print("img_feat = model.encode_image('photo.jpg')")
    print("")
    print("# Example 2: Text search (both encoders)")
    print("model = create_clip_model('ViT-B-16', 'both')")
    print("results = model.search_images('可爱的小猫', image_paths)")
    print("")
    print("# Example 3: Image tagging")
    print("tags = model.tag_image('photo.jpg', ['猫', '狗', '风景'])")
    print("```")
    
    print("\n" + "="*80)
    print("This gives you the best of both worlds!")
    print("="*80 + "\n")
