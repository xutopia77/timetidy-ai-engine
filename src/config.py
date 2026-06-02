"""
AI Engine Configuration
"""

# Model download settings
ALLOW_MODEL_DOWNLOAD = False
MODEL_BASE_URL = "https://hf-mirror.com/timetidy/clip-vision-onnx/resolve/main/clip_vision.onnx"

# Model selection
# Supported models: ViT-B-16, ViT-L-14, ViT-L-14-336, ViT-H-14, RN50
# Default: RN50 (text encoder export works better)
CLIP_MODEL_NAME = "ViT-H-14"

# Model file naming convention
# Vision model: clip_vision_{model_name}.onnx
# Text model: clip_text_{model_name}.onnx
# Tokenizer: tokenizer_{model_name}.json
# Example:
#   CLIP_MODEL_NAME = "ViT-B-16"
#   → clip_vision_ViT_B_16.onnx
#   → clip_text_ViT_B_16.onnx
#   → tokenizer_ViT_B_16.json

# Model feature dimensions (automatically set based on model name)
MODEL_FEATURE_DIMS = {
    "ViT-B-16": 512,
    "ViT-L-14": 768,
    "ViT-L-14-336": 768,
    "ViT-H-14": 1024,
    "RN50": 2048,
}

def get_feature_dim():
    """Get feature dimension for current model"""
    return MODEL_FEATURE_DIMS.get(CLIP_MODEL_NAME, 512)

def get_model_filenames():
    """
    Get model filenames based on configured model name
    
    Returns:
        dict: Dictionary with keys: vision, text, tokenizer
    """
    model_name = CLIP_MODEL_NAME.replace("-", "_").replace("/", "_")
    return {
        "vision": f"clip_vision_{model_name}.onnx",
        "text": f"clip_text_{model_name}.onnx",
        "tokenizer": f"tokenizer_{model_name}.json",
        "pytorch": f"clip_cn_{CLIP_MODEL_NAME.replace('-', '-').replace('/', '-')}.pt",
    }
