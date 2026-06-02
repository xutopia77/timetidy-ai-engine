"""
Chinese-CLIP Model Manager
Download and export Chinese-CLIP models to ONNX format

Features:
- Download Chinese-CLIP models of different sizes
- Export vision encoder to ONNX
- Export text encoder to ONNX
- Support image similarity, text-to-image search, image tagging

Usage:
    # Download all models
    python clip_model_manager.py --download --model all

    # Download specific model
    python clip_model_manager.py --download --model ViT-B-16

    # Export all models to ONNX
    python clip_model_manager.py --export --model all

    # Export specific model to ONNX
    python clip_model_manager.py --export --model ViT-B-16

    # Download and export in one command
    python clip_model_manager.py --download --export --model ViT-B-16

    # Use HuggingFace mirror (faster in China)
    python clip_model_manager.py --download --model ViT-B-16 --mirror
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import torch

try:
    import cn_clip.clip as clip
    from cn_clip.clip import load_from_name
except ImportError:
    print("Error: cn_clip not installed. Please install it first:")
    print("pip install cn-clip")
    sys.exit(1)


HF_MIRRORS = {
    "hf-mirror": "https://hf-mirror.com",
    "aliyun": "https://mirrors.aliyun.com/huggingface",
    "tencent": "https://mirrors.cloud.tencent.com/huggingface",
}


SUPPORTED_MODELS = {
    "ViT-B-16": {
        "description": "Vision Transformer Base, Patch 16 - Balanced performance",
        "params": "86M (vision) + 102M (text)",
        "feature_dim": 512,
        "recommended": True,
    },
    "ViT-L-14": {
        "description": "Vision Transformer Large, Patch 14 - High accuracy",
        "params": "307M (vision) + 102M (text)",
        "feature_dim": 768,
        "recommended": False,
    },
    "ViT-L-14-336": {
        "description": "Vision Transformer Large, Patch 14, 336px - Highest accuracy",
        "params": "307M (vision) + 102M (text)",
        "feature_dim": 768,
        "recommended": False,
    },
    "ViT-H-14": {
        "description": "Vision Transformer Huge, Patch 14 - Research use",
        "params": "632M (vision) + 326M (text)",
        "feature_dim": 1024,
        "recommended": False,
    },
    "RN50": {
        "description": "ResNet-50 - Traditional CNN",
        "params": "39M (vision) + 102M (text)",
        "feature_dim": 2048,
        "recommended": False,
    },
}


def get_model_dir() -> Path:
    """Get the models directory path"""
    script_dir = Path(__file__).parent
    models_dir = script_dir / "models"
    models_dir.mkdir(exist_ok=True)
    return models_dir


def list_models():
    """List all supported models"""
    print("\n" + "=" * 80)
    print("Supported Chinese-CLIP Models")
    print("=" * 80)
    
    for name, info in SUPPORTED_MODELS.items():
        rec = " [RECOMMENDED]" if info["recommended"] else ""
        print(f"\n{name}{rec}")
        print(f"  Description: {info['description']}")
        print(f"  Parameters:  {info['params']}")
        print(f"  Feature Dim: {info['feature_dim']}")
    
    print("\n" + "=" * 80)
    print("Use --model all to download/export all models")
    print("Use --model <name> to download/export specific model")
    print("=" * 80 + "\n")


def download_model(model_name: str, use_mirror: bool = False, mirror_name: str = "hf-mirror") -> Tuple[Path, bool]:
    """
    Download Chinese-CLIP model
    
    Args:
        model_name: Model name (e.g., "ViT-B-16")
        use_mirror: Whether to use HuggingFace mirror
        mirror_name: Mirror name ("hf-mirror", "aliyun", "tencent")
    
    Returns:
        Tuple of (model_path, success)
    """
    models_dir = get_model_dir()
    model_filename = f"clip_cn_{model_name.replace('-', '-').replace('/', '-')}.pt"
    model_path = models_dir / model_filename
    
    print(f"\n{'='*80}")
    print(f"Downloading Chinese-CLIP Model: {model_name}")
    print(f"{'='*80}")
    print(f"Target path: {model_path}")
    
    if model_path.exists():
        print(f"✓ Model already exists: {model_path}")
        return model_path, True
    
    original_endpoint = None
    try:
        if use_mirror:
            if mirror_name not in HF_MIRRORS:
                print(f"Warning: Unknown mirror '{mirror_name}', using 'hf-mirror'")
                mirror_name = "hf-mirror"
            
            mirror_url = HF_MIRRORS[mirror_name]
            original_endpoint = os.environ.get('HF_ENDPOINT')
            os.environ['HF_ENDPOINT'] = mirror_url
            
            print(f"Using HuggingFace mirror: {mirror_url}")
            print(f"This will be faster in China")
        else:
            print(f"Downloading from HuggingFace (official source)...")
            print(f"Tip: Use --mirror for faster download in China")
        
        print(f"This may take several minutes depending on your network speed.")
        
        model, preprocess = load_from_name(model_name, device="cpu")
        
        print(f"✓ Download completed")
        
        torch.save({
            "model_name": model_name,
            "state_dict": model.state_dict(),
            "preprocess": preprocess,
        }, model_path)
        
        file_size = model_path.stat().st_size / (1024 * 1024)
        print(f"✓ Model saved: {model_path}")
        print(f"  File size: {file_size:.1f} MB")
        
        return model_path, True
        
    except Exception as e:
        print(f"✗ Download failed: {e}")
        if use_mirror:
            print(f"  Try without --mirror or try a different mirror")
        else:
            print(f"  Try using --mirror for faster download in China")
        return model_path, False
    finally:
        if original_endpoint is not None:
            os.environ['HF_ENDPOINT'] = original_endpoint
        elif use_mirror and 'HF_ENDPOINT' in os.environ:
            del os.environ['HF_ENDPOINT']


def export_vision_encoder(model_name: str, model) -> Path:
    """
    Export vision encoder to ONNX
    
    Args:
        model_name: Model name
        model: Loaded CLIP model
    
    Returns:
        Path to exported ONNX file
    """
    models_dir = get_model_dir()
    onnx_filename = f"clip_vision_{model_name.replace('-', '_').replace('/', '_')}.onnx"
    onnx_path = models_dir / onnx_filename
    
    print(f"\n--- Exporting Vision Encoder ---")
    print(f"Target: {onnx_path}")
    
    if onnx_path.exists():
        print(f"✓ ONNX file already exists: {onnx_path}")
        return onnx_path
    
    try:
        vision_encoder = model.visual
        vision_encoder.eval()
        
        dummy_input = torch.randn(1, 3, 224, 224)
        
        print(f"Exporting to ONNX...")
        torch.onnx.export(
            vision_encoder,
            dummy_input,
            str(onnx_path),
            input_names=["pixel_values"],
            output_names=["image_features"],
            opset_version=14,
            dynamic_axes={
                "pixel_values": {0: "batch_size"},
                "image_features": {0: "batch_size"}
            }
        )
        
        file_size = onnx_path.stat().st_size / (1024 * 1024)
        print(f"✓ Vision encoder exported: {onnx_path}")
        print(f"  File size: {file_size:.1f} MB")
        
        return onnx_path
        
    except Exception as e:
        print(f"✗ Export failed: {e}")
        raise


def export_text_encoder(model_name: str, model) -> Path:
    """
    Export text encoder to ONNX
    
    Args:
        model_name: Model name
        model: Loaded CLIP model
    
    Returns:
        Path to exported ONNX file
    """
    models_dir = get_model_dir()
    onnx_filename = f"clip_text_{model_name.replace('-', '_').replace('/', '_')}.onnx"
    onnx_path = models_dir / onnx_filename
    
    print(f"\n--- Exporting Text Encoder ---")
    print(f"Target: {onnx_path}")
    
    if onnx_path.exists():
        print(f"✓ ONNX file already exists: {onnx_path}")
        return onnx_path
    
    try:
        text_encoder = model.encode_text
        model.eval()
        
        dummy_text = torch.randint(0, 49408, (1, 77))
        
        print(f"Exporting to ONNX...")
        print(f"Note: Text encoder export may take longer than vision encoder")
        
        class TextEncoderWrapper(torch.nn.Module):
            def __init__(self, clip_model):
                super().__init__()
                self.clip_model = clip_model
            
            def forward(self, input_ids):
                return self.clip_model.encode_text(input_ids)
        
        wrapper = TextEncoderWrapper(model)
        wrapper.eval()
        
        torch.onnx.export(
            wrapper,
            dummy_text,
            str(onnx_path),
            input_names=["input_ids"],
            output_names=["text_features"],
            opset_version=14,
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "text_features": {0: "batch_size"}
            }
        )
        
        file_size = onnx_path.stat().st_size / (1024 * 1024)
        print(f"✓ Text encoder exported: {onnx_path}")
        print(f"  File size: {file_size:.1f} MB")
        
        return onnx_path
        
    except Exception as e:
        print(f"✗ Export failed: {e}")
        print(f"Note: Text encoder export is experimental and may not work for all models")
        raise


def export_tokenizer(model_name: str) -> Path:
    """
    Export tokenizer vocabulary and special tokens
    
    Args:
        model_name: Model name
    
    Returns:
        Path to tokenizer file
    """
    models_dir = get_model_dir()
    tokenizer_filename = f"tokenizer_{model_name.replace('-', '_').replace('/', '_')}.json"
    tokenizer_path = models_dir / tokenizer_filename
    
    print(f"\n--- Exporting Tokenizer ---")
    print(f"Target: {tokenizer_path}")
    
    if tokenizer_path.exists():
        print(f"✓ Tokenizer file already exists: {tokenizer_path}")
        return tokenizer_path
    
    try:
        import json
        
        tokenizer_info = {
            "model_name": model_name,
            "vocab_size": 49408,
            "context_length": 77,
            "sos_token": 49406,
            "eos_token": 49407,
            "pad_token": 0,
            "note": "Use cn_clip.clip.tokenize() for tokenization"
        }
        
        with open(tokenizer_path, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_info, f, indent=2)
        
        print(f"✓ Tokenizer info exported: {tokenizer_path}")
        
        return tokenizer_path
        
    except Exception as e:
        print(f"✗ Export failed: {e}")
        raise


def export_model(model_name: str) -> Dict[str, Path]:
    """
    Export model to ONNX format
    
    Args:
        model_name: Model name
    
    Returns:
        Dictionary of exported file paths
    """
    print(f"\n{'='*80}")
    print(f"Exporting Chinese-CLIP Model: {model_name}")
    print(f"{'='*80}")
    
    exported_files = {}
    
    try:
        print(f"Loading model...")
        model, preprocess = load_from_name(model_name, device="cpu")
        print(f"✓ Model loaded")
        
        vision_path = export_vision_encoder(model_name, model)
        exported_files["vision"] = vision_path
        
        try:
            text_path = export_text_encoder(model_name, model)
            exported_files["text"] = text_path
        except Exception as e:
            print(f"⚠ Text encoder export failed: {e}")
            print(f"  You can still use the model with pre-computed text embeddings")
        
        tokenizer_path = export_tokenizer(model_name)
        exported_files["tokenizer"] = tokenizer_path
        
        print(f"\n{'='*80}")
        print(f"Export Summary for {model_name}")
        print(f"{'='*80}")
        for name, path in exported_files.items():
            size = path.stat().st_size / (1024 * 1024)
            print(f"✓ {name:12s}: {path.name:40s} ({size:6.1f} MB)")
        print(f"{'='*80}\n")
        
        return exported_files
        
    except Exception as e:
        print(f"✗ Export failed: {e}")
        raise


def verify_export(model_name: str) -> bool:
    """
    Verify exported ONNX models work correctly
    
    Args:
        model_name: Model name
    
    Returns:
        True if verification passed
    """
    print(f"\n{'='*80}")
    print(f"Verifying Exported Models: {model_name}")
    print(f"{'='*80}")
    
    models_dir = get_model_dir()
    
    vision_onnx = models_dir / f"clip_vision_{model_name.replace('-', '_').replace('/', '_')}.onnx"
    text_onnx = models_dir / f"clip_text_{model_name.replace('-', '_').replace('/', '_')}.onnx"
    
    try:
        import onnxruntime as ort
        
        if vision_onnx.exists():
            print(f"\n--- Testing Vision Encoder ---")
            session = ort.InferenceSession(str(vision_onnx))
            dummy_image = np.random.randn(1, 3, 224, 224).astype(np.float32)
            output = session.run(None, {"pixel_values": dummy_image})
            print(f"✓ Vision encoder works")
            print(f"  Output shape: {output[0].shape}")
        else:
            print(f"⚠ Vision encoder ONNX not found: {vision_onnx}")
        
        if text_onnx.exists():
            print(f"\n--- Testing Text Encoder ---")
            session = ort.InferenceSession(str(text_onnx))
            dummy_text = np.random.randint(0, 49408, (1, 77)).astype(np.int64)
            output = session.run(None, {"input_ids": dummy_text})
            print(f"✓ Text encoder works")
            print(f"  Output shape: {output[0].shape}")
        else:
            print(f"⚠ Text encoder ONNX not found: {text_onnx}")
        
        print(f"\n✓ Verification completed")
        return True
        
    except ImportError:
        print(f"⚠ onnxruntime not installed, skipping verification")
        print(f"  Install with: pip install onnxruntime")
        return False
    except Exception as e:
        print(f"✗ Verification failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Chinese-CLIP Model Manager - Download and export models to ONNX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # List all supported models
    python clip_model_manager.py --list

    # Download specific model
    python clip_model_manager.py --download --model ViT-B-16

    # Download with mirror (faster in China)
    python clip_model_manager.py --download --model ViT-B-16 --mirror

    # Download all models
    python clip_model_manager.py --download --model all

    # Export specific model to ONNX
    python clip_model_manager.py --export --model ViT-B-16

    # Export all models to ONNX
    python clip_model_manager.py --export --model all

    # Download and export in one command
    python clip_model_manager.py --download --export --model ViT-B-16

    # Verify exported models
    python clip_model_manager.py --verify --model ViT-B-16
        """
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all supported models"
    )
    
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download models"
    )
    
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export models to ONNX format"
    )
    
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify exported ONNX models"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-B-16",
        help="Model name to download/export (default: ViT-B-16). Use 'all' for all models."
    )
    
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Use HuggingFace mirror for faster download in China"
    )
    
    parser.add_argument(
        "--mirror-name",
        type=str,
        default="hf-mirror",
        choices=["hf-mirror", "aliyun", "tencent"],
        help="Mirror name (default: hf-mirror). Options: hf-mirror, aliyun, tencent"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_models()
        return
    
    if not (args.download or args.export or args.verify):
        parser.print_help()
        print("\nError: Please specify at least one action: --download, --export, or --verify")
        return
    
    models_to_process = []
    if args.model.lower() == "all":
        models_to_process = list(SUPPORTED_MODELS.keys())
    elif args.model in SUPPORTED_MODELS:
        models_to_process = [args.model]
    else:
        print(f"Error: Unknown model '{args.model}'")
        print(f"Supported models: {', '.join(SUPPORTED_MODELS.keys())}")
        return
    
    print(f"\n{'='*80}")
    print(f"Chinese-CLIP Model Manager")
    print(f"{'='*80}")
    print(f"Models to process: {', '.join(models_to_process)}")
    print(f"Actions: ", end="")
    actions = []
    if args.download:
        actions.append("Download")
    if args.export:
        actions.append("Export")
    if args.verify:
        actions.append("Verify")
    print(", ".join(actions))
    if args.mirror:
        print(f"Mirror: {args.mirror_name} ({HF_MIRRORS[args.mirror_name]})")
    print(f"{'='*80}")
    
    for model_name in models_to_process:
        if args.download:
            model_path, success = download_model(
                model_name,
                use_mirror=args.mirror,
                mirror_name=args.mirror_name
            )
            if not success:
                print(f"✗ Skipping {model_name} due to download failure")
                continue
        
        if args.export:
            try:
                export_model(model_name)
            except Exception as e:
                print(f"✗ Export failed for {model_name}: {e}")
                continue
        
        if args.verify:
            verify_export(model_name)
    
    print(f"\n{'='*80}")
    print(f"All operations completed!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
