"""
Export Chinese-CLIP model to ONNX format and pre-compute text embeddings.

Usage:
    1. Install dependencies (one-time):
       pip install torch torchvision cn_clip onnx onnxsim onnxruntime

    2. Set mirror if needed (China users):
       set HF_ENDPOINT=https://hf-mirror.com

    3. Run this script:
       cd time-tidy/src/ai_engine
       python export_clip_onnx.py

This script will:
    1. Download Chinese-CLIP ViT-B-16 model (~1.7GB)
    2. Export vision encoder to ONNX (~340MB)
    3. Compute text embeddings for tag_vocab.json
    4. Save all files to models/ directory
"""

import os
import sys
import json


def check_dependencies():
    missing = []
    try:
        import torch
    except ImportError:
        missing.append("torch")

    try:
        import cn_clip
    except ImportError:
        missing.append("cn_clip")

    try:
        import onnxruntime
    except ImportError:
        missing.append("onnxruntime")

    if missing:
        print("Missing dependencies:", ", ".join(missing))
        print("Install with: pip install torch torchvision cn_clip onnxruntime onnx onnxsim")
        sys.exit(1)


def main():
    check_dependencies()

    import torch
    import numpy as np
    import cn_clip.clip as clip

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, "models")
    vocab_path = os.path.join(model_dir, "tag_vocab.json")
    vision_onnx_path = os.path.join(model_dir, "clip_vision.onnx")
    embeddings_path = os.path.join(model_dir, "tag_embeddings.npz")
    pretrained_dir = os.path.join(model_dir, "pretrained_weights")

    os.makedirs(model_dir, exist_ok=True)

    if not os.path.exists(vocab_path):
        print(f"Error: tag vocab not found: {vocab_path}")
        print("Please create tag_vocab.json first.")
        sys.exit(1)

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
        tags = vocab["tags"]

    print(f"Loaded {len(tags)} tags from vocab")

    print("Loading Chinese-CLIP ViT-B-16 model...")
    print("(This will download ~1.7GB model on first run)")
    if os.environ.get("HF_ENDPOINT"):
        print(f"Using HF mirror: {os.environ['HF_ENDPOINT']}")

    model, preprocess = clip.load_from_name(
        "ViT-B-16",
        device="cpu",
        download_root=pretrained_dir,
    )
    model.eval()
    print("Model loaded successfully")

    print("\nExporting vision encoder to ONNX...")
    dummy_image = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        torch.onnx.export(
            model.visual,
            dummy_image,
            vision_onnx_path,
            input_names=["pixel_values"],
            output_names=["image_features"],
            opset_version=14,
            do_constant_folding=True,
        )

    onnx_size_mb = os.path.getsize(vision_onnx_path) / (1024 * 1024)
    print(f"Vision encoder saved: {vision_onnx_path} ({onnx_size_mb:.1f} MB)")

    try:
        import onnx
        import onnxsim

        print("Simplifying ONNX model...")
        onnx_model = onnx.load(vision_onnx_path)
        simplified_model, check = onnxsim.simplify(onnx_model)
        if check:
            onnx.save(simplified_model, vision_onnx_path)
            new_size_mb = os.path.getsize(vision_onnx_path) / (1024 * 1024)
            print(f"Simplified: {onnx_size_mb:.1f} MB -> {new_size_mb:.1f} MB")
        else:
            print("Simplification check failed, keeping original model")
    except ImportError:
        print("onnxsim not installed, skipping simplification")

    print("\nComputing text embeddings with prompt templates...")

    PROMPT_TEMPLATES = [
        "一张{}的照片",
        "{}的图片",
        "{}的场景",
    ]

    batch_size = 32
    tag_embeddings = np.zeros((len(tags), 512), dtype=np.float32)

    for tmpl_idx, template in enumerate(PROMPT_TEMPLATES):
        print(f"  Template {tmpl_idx + 1}/{len(PROMPT_TEMPLATES)}: {template}")
        prompted_tags = [template.format(tag) for tag in tags]
        all_features = []

        for i in range(0, len(prompted_tags), batch_size):
            batch_tags = prompted_tags[i : i + batch_size]
            text_tokens = clip.tokenize(batch_tags)

            with torch.no_grad():
                text_features = model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                all_features.append(text_features.numpy())

        template_embeddings = np.concatenate(all_features, axis=0)
        tag_embeddings += template_embeddings

    tag_embeddings = tag_embeddings / len(PROMPT_TEMPLATES)
    tag_embeddings = tag_embeddings / np.linalg.norm(tag_embeddings, axis=-1, keepdims=True)

    logit_scale = model.logit_scale.exp().item()
    print(f"Model logit_scale: {logit_scale:.4f}")

    np.savez(
        embeddings_path,
        embeddings=tag_embeddings,
        tags=np.array(tags),
        logit_scale=np.array([logit_scale]),
    )

    emb_size_kb = os.path.getsize(embeddings_path) / 1024
    print(f"Text embeddings saved: {embeddings_path} ({emb_size_kb:.1f} KB)")
    print(f"Embedding shape: {tag_embeddings.shape}")
    print(f"Prompt templates used: {PROMPT_TEMPLATES}")

    print("\nVerifying ONNX model...")
    import onnxruntime as ort

    session = ort.InferenceSession(vision_onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: dummy_image.numpy()})
    print(f"ONNX output shape: {output[0].shape}")

    cosine_sim = np.dot(tag_embeddings[0], tag_embeddings[0])
    print(f"Text embedding norm check (should be ~1.0): {cosine_sim:.6f}")

    print("\n" + "=" * 60)
    print("Export completed successfully!")
    print("=" * 60)
    print(f"  Vision model:  {vision_onnx_path}")
    print(f"  Tag embeddings: {embeddings_path}")
    print(f"  Tag vocab:     {vocab_path}")
    print()
    print(f"You can safely delete pretrained weights: {pretrained_dir}")
    print("Runtime only needs: onnxruntime, Pillow, numpy")


if __name__ == "__main__":
    main()
