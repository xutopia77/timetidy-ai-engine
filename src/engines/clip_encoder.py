"""
CLIP Encoder for feature extraction

This module provides:
1. Image feature extraction (Vision Encoder)
2. Text feature extraction (Text Encoder)
3. ONNX model loading and inference
"""

import os
import numpy as np
from PIL import Image
import onnxruntime as ort
from pathlib import Path

import config
from utils.logger import log


MODEL_DIR = Path(__file__).parent.parent / "models"

CLIP_IMAGE_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_IMAGE_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


class CLIPEncoder:
    def __init__(self):
        self._vision_session = None
        self._text_session = None
        self._logit_scale = 14.29
        
        model_filenames = config.get_model_filenames()
        self._vision_model_path = MODEL_DIR / model_filenames["vision"]
        self._text_model_path = MODEL_DIR / model_filenames["text"]
        
        log.info(f"CLIP Encoder initialized:")
        log.info(f"  - Model: {config.CLIP_MODEL_NAME}")
        log.info(f"  - Vision model: {model_filenames['vision']}")
        log.info(f"  - Text model: {model_filenames['text']}")
    
    def _ensure_model(self):
        if self._vision_session is not None:
            return
        
        if not self._vision_model_path.exists():
            raise FileNotFoundError(
                f"Vision model not found: {self._vision_model_path}\n"
                f"Please run: python scripts/clip_model_manager.py --export --model {config.CLIP_MODEL_NAME}"
            )
        
        if not self._text_model_path.exists():
            raise FileNotFoundError(
                f"Text model not found: {self._text_model_path}\n"
                f"Please run: python scripts/clip_model_manager.py --export --model {config.CLIP_MODEL_NAME}"
            )
        
        log.info(f"Loading CLIP models...")
        
        self._vision_session = ort.InferenceSession(
            str(self._vision_model_path),
            providers=["CPUExecutionProvider"],
        )
        
        self._text_session = ort.InferenceSession(
            str(self._text_model_path),
            providers=["CPUExecutionProvider"],
        )
        
        log.info(
            f"CLIP models loaded: "
            f"Vision encoder ready,"
            f"Text encoder ready,"
            f"Feature dim: {config.get_feature_dim()}"
        )
    
    def _preprocess_image(self, image_path: str) -> np.ndarray:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((224, 224), Image.BICUBIC)
        img_array = np.array(img, dtype=np.float32)
        img_array = (img_array / 255.0 - CLIP_IMAGE_MEAN) / CLIP_IMAGE_STD
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, 0)
        return img_array
    
    def _tokenize_text(self, text: str) -> np.ndarray:
        try:
            import cn_clip.clip as clip
            tokens = clip.tokenize([text])
            tokens_array = tokens.numpy()
            
            if tokens_array.shape[1] != 77:
                if tokens_array.shape[1] < 77:
                    pad_length = 77 - tokens_array.shape[1]
                    padding = np.zeros((tokens_array.shape[0], pad_length), dtype=tokens_array.dtype)
                    tokens_array = np.concatenate([tokens_array, padding], axis=1)
                else:
                    tokens_array = tokens_array[:, :77]
            
            return tokens_array
        except ImportError:
            raise ImportError(
                "cn_clip not installed. Please install it:\n"
                "pip install cn-clip"
            )
    
    def encode_image(self, image_path: str) -> np.ndarray:
        self._ensure_model()
        
        input_data = self._preprocess_image(image_path)
        input_name = self._vision_session.get_inputs()[0].name
        features = self._vision_session.run(None, {input_name: input_data})[0]
        
        features = features / np.linalg.norm(features, axis=-1, keepdims=True)
        return features
    
    def encode_text(self, text: str) -> np.ndarray:
        self._ensure_model()
        
        tokens = self._tokenize_text(text)
        input_name = self._text_session.get_inputs()[0].name
        features = self._text_session.run(None, {input_name: tokens})[0]
        
        features = features / np.linalg.norm(features, axis=-1, keepdims=True)
        return features
    
    def compute_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        return float(np.dot(features1, features2.T)[0][0])
    
    def get_status(self) -> dict:
        return {
            "models_loaded": self._vision_session is not None,
            "vision_model": str(self._vision_model_path),
            "text_model": str(self._text_model_path),
            "feature_dim": config.get_feature_dim()
        }