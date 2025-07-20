import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.clip_wrapper import CLIPWrapper
from multiview_dataset import MultiViewDataset
from utils import load_clip_preprocessors


def train(
    jsonl_path="./data/pix3d/usable_samples.jsonl",
    image_dir="./data/pix3d/renders",
    batch_size=32,
    epochs=10,
    lr=1e-5,
    fine_tune="full",  # "linear" hoặc "full"
    save_path="./weights/clip_finetune_full.pt",
    log_dir="./logs/full"
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load CLIP + preprocess
    base_model, preprocess_fn, tokenizer_fn = load_clip_preprocessors()
    model = CLIPWrapper(base_model, train_mode=fine_tune).to(device)

    # Dataset & Dataloader
    dataset = MultiViewDataset(jsonl_path, image_dir, preprocess_fn)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Optimizer: Tối ưu toàn bộ mô hình
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CosineEmbeddingLoss()

    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for images, captions in dataloader:
            images = images.to(device)
            text_tokens = tokenizer_fn(captions).to(device)

            image_features = model.encode_image(images)
            text_features = model.encode_text(text_tokens)

            targets = torch.ones(images.size(0)).to(device)
            loss = criterion(image_features, text_features, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"🟦 [Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.4f}")
        writer.add_scalar("Train/Loss", avg_loss, epoch + 1)

        # Save checkpoint mỗi epoch (tuỳ chọn)
        checkpoint_path = os.path.join(os.path.dirname(save_path), f"epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), checkpoint_path)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"✅ Finished training. Final model saved to {save_path}")


if __name__ == "__main__":
    train()
