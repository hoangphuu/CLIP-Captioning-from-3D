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
    fine_tune="linear",  # "linear" or "full"
    save_path="./weights/clip_finetune.pt",
    log_dir="./logs"
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model, preprocess, tokenizer = load_clip_preprocessors()
    model = CLIPWrapper(base_model, train_mode=fine_tune).to(device)

    dataset = MultiViewDataset(jsonl_path, image_dir, preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CosineEmbeddingLoss()
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for images, captions in dataloader:
            images = images.to(device)
            tokens = tokenizer(captions).to(device)
            image_feats = model.encode_image(images)
            text_feats = model.encode_text(tokens)
            targets = torch.ones(images.size(0)).to(device)

            loss = criterion(image_feats, text_feats, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"📘 Epoch {epoch+1}, Loss: {avg_loss:.4f}")
        writer.add_scalar("Train/Loss", avg_loss, epoch)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"✅ Saved model to {save_path}")


if __name__ == "__main__":
    train()

