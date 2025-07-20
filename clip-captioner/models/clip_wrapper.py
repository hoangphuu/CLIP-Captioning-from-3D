import torch
import torch.nn as nn
import clip  # OpenAI CLIP

class CLIPWrapper(nn.Module):
    def __init__(self, base_model, train_mode="linear"):
        """
        Wrapper để kiểm soát fine-tune toàn bộ hoặc chỉ linear layer.

        Args:
            base_model (nn.Module): CLIP model từ openai/clip.
            train_mode (str): "linear" hoặc "full".
        """
        super().__init__()
        self.clip_model = base_model
        self.train_mode = train_mode.lower()

        if self.train_mode == "linear":
            self.freeze_all_but_projection()
        elif self.train_mode == "full":
            self.unfreeze_all()
        else:
            raise ValueError(f"Invalid train_mode: {train_mode}. Use 'linear' or 'full'.")

    def freeze_all_but_projection(self):
        """Đóng băng toàn bộ CLIP, chỉ cho phép fine-tune các projection head."""
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Projection layers được fine-tune
        for name, param in self.clip_model.visual.named_parameters():
            if "proj" in name:
                param.requires_grad = True
        for name, param in self.clip_model.transformer.named_parameters():
            if "proj" in name:
                param.requires_grad = True
        for param in self.clip_model.text_projection.parameters():
            param.requires_grad = True
        for param in self.clip_model.logit_scale.parameters():
            param.requires_grad = True

    def unfreeze_all(self):
        """Fine-tune toàn bộ CLIP."""
        for param in self.clip_model.parameters():
            param.requires_grad = True

    def encode_image(self, images):
        """
        Encode ảnh → embedding CLIP.
        Args:
            images: Tensor (B, 3, H, W)
        Returns:
            Tensor (B, D)
        """
        return self.clip_model.encode_image(images)

    def encode_text(self, text_tokens):
        """
        Encode caption → embedding CLIP.
        Args:
            text_tokens: Tensor (B, seq_len) từ clip.tokenize()
        Returns:
            Tensor (B, D)
        """
        return self.clip_model.encode_text(text_tokens)

    def forward(self, images, text_tokens):
        """
        Encode ảnh và caption cùng lúc.
        Returns:
            Tuple[Tensor, Tensor]: (image_features, text_features)
        """
        image_features = self.encode_image(images)
        text_features = self.encode_text(text_tokens)
        return image_features, text_features
