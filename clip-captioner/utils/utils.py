import os
import torch
from PIL import Image
import clip


def load_clip_preprocessors(model_name="ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Load the CLIP model, tokenizer, and preprocessing transforms.
    
    Returns:
        model: CLIP model
        preprocess: image preprocessing function
        tokenizer: function to tokenize text
    """
    model, preprocess = clip.load(model_name, device=device)
    tokenizer = clip.tokenize
    return model, preprocess, tokenizer


def preprocess_image(image_path, preprocess_fn):
    """
    Load and preprocess image for CLIP.

    Args:
        image_path (str): Path to image file
        preprocess_fn: preprocessing function from CLIP (e.g., resize, normalize)

    Returns:
        Tensor: preprocessed image tensor (1, 3, H, W)
    """
    image = Image.open(image_path).convert("RGB")
    return preprocess_fn(image)


def preprocess_caption(caption, tokenizer_fn):
    """
    Preprocess caption for CLIP.

    Args:
        caption (str): Input caption
        tokenizer_fn: tokenizer from CLIP

    Returns:
        Tensor: tokenized caption (1, seq_len)
    """
    return tokenizer_fn([caption], truncate=True)[0]


def load_clip_preprocessors(clip_model_name="ViT-B/32", device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(clip_model_name, device=device)

    def tokenize_fn(text_list):
        return clip.tokenize(text_list).to(device)

    return model, preprocess, tokenize_fn
