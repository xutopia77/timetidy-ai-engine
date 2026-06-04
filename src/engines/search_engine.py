"""
Search Engine using Chinese-CLIP ONNX models

This engine provides:
1. Image feature extraction
2. Text feature extraction
3. Text-to-image search
4. Image-to-image similarity
"""

import os
import sys
import numpy as np
from PIL import Image
import onnxruntime as ort
from engines.base import BaseEngine
import config

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

CLIP_IMAGE_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_IMAGE_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


class SearchEngine(BaseEngine):
    def __init__(self):
        self._vision_session = None
        self._text_session = None
        self._logit_scale = 14.29
        
        model_filenames = config.get_model_filenames()
        self._vision_model_path = os.path.join(MODEL_DIR, model_filenames["vision"])
        self._text_model_path = os.path.join(MODEL_DIR, model_filenames["text"])
        
        print(f"Using model: {config.CLIP_MODEL_NAME}", file=sys.stderr)
        print(f"Vision model: {model_filenames['vision']}", file=sys.stderr)
        print(f"Text model: {model_filenames['text']}", file=sys.stderr)

    def _ensure_model(self):
        if self._vision_session is not None:
            return

        os.makedirs(MODEL_DIR, exist_ok=True)

        if not os.path.exists(self._vision_model_path):
            if config.ALLOW_MODEL_DOWNLOAD:
                self._download_vision_model()
            else:
                raise FileNotFoundError(
                    f"Vision model not found: {self._vision_model_path}\n"
                    f"Please run: python clip_model_manager.py --export --model {config.CLIP_MODEL_NAME}\n"
                    f"Or set ALLOW_MODEL_DOWNLOAD=true in config.py"
                )

        if not os.path.exists(self._text_model_path):
            raise FileNotFoundError(
                f"Text model not found: {self._text_model_path}\n"
                f"Please run: python clip_model_manager.py --export --model {config.CLIP_MODEL_NAME}"
            )

        self._vision_session = ort.InferenceSession(
            self._vision_model_path,
            providers=["CPUExecutionProvider"],
        )

        self._text_session = ort.InferenceSession(
            self._text_model_path,
            providers=["CPUExecutionProvider"],
        )

        print(
            f"CLIP search engine ready:\n"
            f"  - Vision encoder loaded\n"
            f"  - Text encoder loaded\n"
            f"  - Feature dim: {config.get_feature_dim()}",
            file=sys.stderr,
        )

    def _download_vision_model(self):
        import urllib.request

        base_url = os.environ.get("MODEL_BASE_URL", config.MODEL_BASE_URL)
        if base_url.endswith("/"):
            url = base_url + os.path.basename(self._vision_model_path)
        else:
            url = base_url

        print(f"Downloading CLIP vision model from {url} ...", file=sys.stderr)
        tmp_path = self._vision_model_path + ".tmp"
        try:
            urllib.request.urlretrieve(url, tmp_path)
            os.rename(tmp_path, self._vision_model_path)
            print(f"Model saved to {self._vision_model_path}", file=sys.stderr)
        except Exception as e:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise RuntimeError(
                f"Failed to download CLIP vision model: {e}\n"
                f"Please run: python clip_model_manager.py --export --model {config.CLIP_MODEL_NAME}"
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

    def process(self, params: dict) -> dict:
        self._ensure_model()

        action = params.get("action", "")
        
        if action == "encode_image":
            file_path = params.get("file_path", "")
            if not file_path:
                raise ValueError("file_path is required")
            
            features = self.encode_image(file_path)
            return {
                "features": features.tolist(),
                "shape": features.shape,
            }
        
        elif action == "encode_text":
            text = params.get("text", "")
            if not text:
                raise ValueError("text is required")
            
            features = self.encode_text(text)
            return {
                "features": features.tolist(),
                "shape": features.shape,
            }
        
        elif action == "search_by_text":
            text = params.get("text", "")
            file_paths = params.get("file_paths", [])
            top_k = params.get("top_k", 5)
            
            if not text:
                raise ValueError("text is required")
            if not file_paths:
                raise ValueError("file_paths is required")
            
            text_features = self.encode_text(text)
            
            results = []
            for fp in file_paths:
                try:
                    img_features = self.encode_image(fp)
                    similarity = self.compute_similarity(text_features, img_features)
                    results.append({
                        "file_path": fp,
                        "similarity": round(similarity, 4),
                    })
                except Exception as e:
                    print(f"Warning: Failed to process {fp}: {e}", file=sys.stderr)
            
            results.sort(key=lambda x: x["similarity"], reverse=True)
            return {
                "query": text,
                "results": results[:top_k],
            }
        
        elif action == "compare_images":
            file_path1 = params.get("file_path1", "")
            file_path2 = params.get("file_path2", "")
            
            if not file_path1 or not file_path2:
                raise ValueError("file_path1 and file_path2 are required")
            
            features1 = self.encode_image(file_path1)
            features2 = self.encode_image(file_path2)
            similarity = self.compute_similarity(features1, features2)
            
            return {
                "file_path1": file_path1,
                "file_path2": file_path2,
                "similarity": round(similarity, 4),
            }
        
        else:
            raise ValueError(f"Unknown action: {action}")