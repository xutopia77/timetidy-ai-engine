"""
AI Engine Configuration

All configuration must be set by Go program via cfg_set command.
No default values for paths - ensures single source of truth.
"""

import os
from pathlib import Path
from typing import Optional


# ============================================================================
# Model Auto-Detection
# ============================================================================

# Model size tiers: priority order H > M > L
MODEL_TIERS = ["H", "M", "L"]

# Known model name patterns for each tier
TIER_MODEL_NAMES = {
    "H": "ViT-H-14",
    "M": "ViT-L-14",
    "L": "ViT-B-16",
}

# Model feature dimensions
MODEL_FEATURE_DIMS = {
    "ViT-B-16": 512,
    "ViT-L-14": 768,
    "ViT-L-14-336": 768,
    "ViT-H-14": 1024,
    "RN50": 2048,
}

# Model download settings
ALLOW_MODEL_DOWNLOAD = False
MODEL_BASE_URL = "https://hf-mirror.com/timetidy/clip-vision-onnx/resolve/main/clip_vision.onnx"


def detect_models(models_root: str) -> dict:
    """
    Auto-detect available model tiers under models_root.
    
    Scans models_root/H, models_root/M, models_root/L directories
    for ONNX model files (clip_vision_*.onnx).
    
    Returns:
        dict: {"tier": "H"|"M"|"L", "model_name": str, "model_dir": Path}
        None if no models found
    """
    for tier in MODEL_TIERS:
        tier_dir = Path(models_root) / tier
        if not tier_dir.exists():
            continue
        
        # Check if vision model exists in this tier
        for f in tier_dir.iterdir():
            if f.name.startswith("clip_vision_") and f.name.endswith(".onnx"):
                model_name = TIER_MODEL_NAMES.get(tier, "unknown")
                return {
                    "tier": tier,
                    "model_name": model_name,
                    "model_dir": tier_dir,
                }
    
    return None


# ============================================================================
# Runtime Configuration (must be set by Go program)
# ============================================================================

class RuntimeConfig:
    """Runtime configuration passed from Go program"""
    
    def __init__(self):
        # Paths (all must be set by external program)
        self.progress_file: Optional[str] = None
        self.index_dir: Optional[str] = None
        self.tag_config_file: Optional[str] = None
        self.prompt_cache_dir: Optional[str] = None
        self.models_root: Optional[str] = None  # ai_engine/models (parent of H/M/L)
        
        # Log
        self.log_level: str = "INFO"
        self.log_file: Optional[str] = None
        
        # Auto-detected model info (set during init, not from config)
        self._detected_tier: Optional[str] = None
        self._detected_model_name: Optional[str] = None
        self._detected_model_dir: Optional[Path] = None
    
    def load_from_dict(self, config_dict: dict):
        """Load config from dictionary (from Go program)"""
        if "progress_file" in config_dict:
            self.progress_file = config_dict["progress_file"]
        if "index_dir" in config_dict:
            self.index_dir = config_dict["index_dir"]
        if "tag_config_file" in config_dict:
            self.tag_config_file = config_dict["tag_config_file"]
        if "prompt_cache_dir" in config_dict:
            self.prompt_cache_dir = config_dict["prompt_cache_dir"]
        if "models_root" in config_dict:
            self.models_root = config_dict["models_root"]
        if "log_level" in config_dict:
            self.log_level = config_dict["log_level"]
        if "log_file" in config_dict:
            self.log_file = config_dict["log_file"]
    
    def auto_detect_model(self) -> bool:
        """Auto-detect best available model tier. Returns True if found."""
        if not self.models_root:
            return False
        
        result = detect_models(self.models_root)
        if result:
            self._detected_tier = result["tier"]
            self._detected_model_name = result["model_name"]
            self._detected_model_dir = result["model_dir"]
            return True
        return False
    
    def is_configured(self) -> bool:
        """Check if all required paths are configured"""
        return all([
            self.progress_file,
            self.index_dir,
            self.tag_config_file,
            self.prompt_cache_dir,
            self.models_root,
        ])
    
    def get_missing_config(self) -> list:
        """Get list of missing required configurations"""
        missing = []
        if not self.progress_file:
            missing.append("progress_file")
        if not self.index_dir:
            missing.append("index_dir")
        if not self.tag_config_file:
            missing.append("tag_config_file")
        if not self.prompt_cache_dir:
            missing.append("prompt_cache_dir")
        if not self.models_root:
            missing.append("models_root")
        return missing


# Global runtime config instance
RUNTIME_CONFIG = RuntimeConfig()


# ============================================================================
# Helper Functions
# ============================================================================

def get_clip_model_name() -> str:
    """Get CLIP model name (auto-detected)"""
    if RUNTIME_CONFIG._detected_model_name:
        return RUNTIME_CONFIG._detected_model_name
    return "unknown"


def get_feature_dim() -> int:
    """Get feature dimension for current model"""
    model_name = get_clip_model_name()
    return MODEL_FEATURE_DIMS.get(model_name, 512)


def get_model_filenames() -> dict:
    """
    Get model filenames based on detected model name
    
    Returns:
        dict: Dictionary with keys: vision, text, tokenizer
    """
    model_name = get_clip_model_name().replace("-", "_").replace("/", "_")
    return {
        "vision": f"clip_vision_{model_name}.onnx",
        "text": f"clip_text_{model_name}.onnx",
        "tokenizer": f"tokenizer_{model_name}.json",
    }


def get_progress_file() -> Path:
    """Get progress file path (must be configured)"""
    if not RUNTIME_CONFIG.progress_file:
        raise ValueError("progress_file not configured. Please call cfg_set first.")
    return Path(RUNTIME_CONFIG.progress_file)


def get_index_dir() -> str:
    """Get index directory (must be configured)"""
    if not RUNTIME_CONFIG.index_dir:
        raise ValueError("index_dir not configured. Please call cfg_set first.")
    return RUNTIME_CONFIG.index_dir


def get_tag_config_file() -> Path:
    """Get tag config file path (must be configured)"""
    if not RUNTIME_CONFIG.tag_config_file:
        raise ValueError("tag_config_file not configured. Please call cfg_set first.")
    return Path(RUNTIME_CONFIG.tag_config_file)


def get_prompt_cache_dir() -> Path:
    """Get prompt cache directory (must be configured)"""
    if not RUNTIME_CONFIG.prompt_cache_dir:
        raise ValueError("prompt_cache_dir not configured. Please call cfg_set first.")
    return Path(RUNTIME_CONFIG.prompt_cache_dir)


def get_model_dir() -> Path:
    """Get model directory (auto-detected from models_root/H|M|L)"""
    if RUNTIME_CONFIG._detected_model_dir:
        return RUNTIME_CONFIG._detected_model_dir
    raise ValueError("No model detected. Please ensure models are placed in models/H, models/M, or models/L directory.")


def get_model_tier() -> str:
    """Get detected model tier (H/M/L)"""
    if RUNTIME_CONFIG._detected_tier:
        return RUNTIME_CONFIG._detected_tier
    return "unknown"
