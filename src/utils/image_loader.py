from PIL import Image
import os


def load_image(image_path: str) -> Image.Image:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image
