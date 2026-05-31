import os
import sys
import json
import numpy as np
from PIL import Image
import onnxruntime as ort
from engines.base import BaseEngine

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
VISION_MODEL_FILENAME = "clip_vision.onnx"
VOCAB_FILENAME = "tag_vocab.json"
EMBEDDINGS_FILENAME = "tag_embeddings.npz"
DEFAULT_MODEL_URL = "https://hf-mirror.com/timetidy/clip-vision-onnx/resolve/main/clip_vision.onnx"

CLIP_IMAGE_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_IMAGE_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


class TagEngine(BaseEngine):
    def __init__(self):
        self._vision_session = None
        self._text_embeddings = None
        self._vocab_tags = []
        self._logit_scale = 14.29
        self._vision_model_path = os.path.join(MODEL_DIR, VISION_MODEL_FILENAME)
        self._vocab_path = os.path.join(MODEL_DIR, VOCAB_FILENAME)
        self._embeddings_path = os.path.join(MODEL_DIR, EMBEDDINGS_FILENAME)

    def _ensure_model(self):
        if self._vision_session is not None:
            return

        os.makedirs(MODEL_DIR, exist_ok=True)

        if not os.path.exists(self._vocab_path):
            raise FileNotFoundError(
                f"Tag vocab file not found: {self._vocab_path}\n"
                f"Please run: python export_clip_onnx.py"
            )

        if not os.path.exists(self._embeddings_path):
            raise FileNotFoundError(
                f"Tag embeddings file not found: {self._embeddings_path}\n"
                f"Please run: python export_clip_onnx.py"
            )

        if not os.path.exists(self._vision_model_path):
            self._download_model()

        self._vision_session = ort.InferenceSession(
            self._vision_model_path,
            providers=["CPUExecutionProvider"],
        )

        with open(self._vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
            self._vocab_tags = vocab["tags"]

        data = np.load(self._embeddings_path, allow_pickle=True)
        self._text_embeddings = data["embeddings"]

        if "logit_scale" in data:
            self._logit_scale = float(data["logit_scale"][0])

        if len(self._vocab_tags) != self._text_embeddings.shape[0]:
            raise ValueError(
                f"Tag vocab ({len(self._vocab_tags)} tags) and embeddings "
                f"({self._text_embeddings.shape[0]} entries) mismatch. "
                f"Please re-run: python export_clip_onnx.py"
            )

        print(
            f"CLIP tag engine ready: {len(self._vocab_tags)} tags, "
            f"embedding dim={self._text_embeddings.shape[1]}, "
            f"logit_scale={self._logit_scale:.2f}",
            file=sys.stderr,
        )

    def _download_model(self):
        import urllib.request

        base_url = os.environ.get("MODEL_BASE_URL", DEFAULT_MODEL_URL)
        if base_url.endswith("/"):
            url = base_url + VISION_MODEL_FILENAME
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
                f"Please run: python export_clip_onnx.py"
            )

    def _preprocess(self, image_path: str) -> np.ndarray:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((224, 224), Image.BICUBIC)
        img_array = np.array(img, dtype=np.float32)
        img_array = (img_array / 255.0 - CLIP_IMAGE_MEAN) / CLIP_IMAGE_STD
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, 0)
        return img_array

    def process(self, params: dict) -> dict:
        self._ensure_model()

        image_path = params.get("image_path", "")
        top_k = params.get("top_k", 5)
        threshold = params.get("threshold", 0.05)

        if not image_path:
            raise ValueError("image_path is required")

        input_data = self._preprocess(image_path)
        input_name = self._vision_session.get_inputs()[0].name
        image_features = self._vision_session.run(None, {input_name: input_data})[0]

        image_features = image_features / np.linalg.norm(
            image_features, axis=-1, keepdims=True
        )

        logits = self._logit_scale * (image_features @ self._text_embeddings.T)[0]
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / exp_logits.sum()

        top_indices = np.argsort(probabilities)[-top_k:][::-1]

        tags = []
        for idx in top_indices:
            score = float(probabilities[idx])
            if score >= threshold:
                tags.append(
                    {
                        "label": self._vocab_tags[idx],
                        "confidence": round(score, 4),
                    }
                )

        return {"tags": tags}
