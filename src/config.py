"""
AI Engine Configuration

All configuration must be set by Go program via cfg_set command.
No default values for paths - ensures single source of truth.
"""

import os
from pathlib import Path
from typing import Optional


# ============================================================================
# Runtime Configuration (must be set by Go program)
# ============================================================================

class RuntimeConfig:
    """Runtime configuration passed from Go program"""
    
    def __init__(self):
        # Paths (all must be set by external program)
        self.progress_file: Optional[str] = None # ai-data/progress/process_file.json
        self.index_dir: Optional[str] = None # ai-data/index
        self.tag_config_file: Optional[str] = None # src/config/hierarchical_tags.json
        self.prompt_cache_dir: Optional[str] = None # ai-data/prompt_cache
        self.model_dir: Optional[str] = None # src/models/clip_model
        
        # Model
        self.clip_model_name: str = "ViT-H-14"
        
        # Log
        self.log_level: str = "INFO"
        self.log_file: Optional[str] = None
    
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
        if "model_dir" in config_dict:
            self.model_dir = config_dict["model_dir"]
        if "clip_model_name" in config_dict:
            self.clip_model_name = config_dict["clip_model_name"]
        if "log_level" in config_dict:
            self.log_level = config_dict["log_level"]
        if "log_file" in config_dict:
            self.log_file = config_dict["log_file"]
    
    def is_configured(self) -> bool:
        """Check if all required paths are configured"""
        return all([
            self.progress_file,
            self.index_dir,
            self.tag_config_file,
            self.prompt_cache_dir,
            self.model_dir,
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
        if not self.model_dir:
            missing.append("model_dir")
        return missing


# Global runtime config instance
RUNTIME_CONFIG = RuntimeConfig()


# ============================================================================
# Model Configuration
# ============================================================================

# Model download settings
ALLOW_MODEL_DOWNLOAD = False
MODEL_BASE_URL = "https://hf-mirror.com/timetidy/clip-vision-onnx/resolve/main/clip_vision.onnx"

# Model feature dimensions (automatically set based on model name)
MODEL_FEATURE_DIMS = {
    "ViT-B-16": 512,
    "ViT-L-14": 768,
    "ViT-L-14-336": 768,
    "ViT-H-14": 1024,
    "RN50": 2048,
}


# ============================================================================
# Helper Functions
# ============================================================================

def get_clip_model_name() -> str:
    """Get CLIP model name"""
    return RUNTIME_CONFIG.clip_model_name


def get_feature_dim() -> int:
    """Get feature dimension for current model"""
    model_name = get_clip_model_name()
    return MODEL_FEATURE_DIMS.get(model_name, 512)


def get_model_filenames() -> dict:
    """
    Get model filenames based on configured model name
    
    Returns:
        dict: Dictionary with keys: vision, text, tokenizer
    """
    model_name = get_clip_model_name().replace("-", "_").replace("/", "_")
    return {
        "vision": f"clip_vision_{model_name}.onnx",
        "text": f"clip_text_{model_name}.onnx",
        "tokenizer": f"tokenizer_{model_name}.json",
        "pytorch": f"clip_cn_{get_clip_model_name().replace('-', '-').replace('/', '-')}.pt",
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
    """Get model directory (must be configured)"""
    if not RUNTIME_CONFIG.model_dir:
        raise ValueError("model_dir not configured. Please call cfg_set first.")
    return Path(RUNTIME_CONFIG.model_dir)
