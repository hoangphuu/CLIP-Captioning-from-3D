import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import clip
from multiview_dataset import MultiViewDataset
from torch.utils.tensorboard import SummaryWriter


class LinearHead(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        return self.linear(x)


def train_clip_linear(
    jsonl_path="./data/pix3d/usable_samples.jsonl",
    image_dir="./data/pix3d/renders",
    batch_size=32,
    epochs=10,
    lr=1e-4,
    save_path="./weights/linear_clip.pt",
    log_dir="./logs_linear"
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    dataset = MultiViewDataset(jsonl_path, image_dir, preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    head = LinearHead(model.visual.output_dim).to(device)

    optimizer = optim.Adam(head.parameters(), lr=lr)
    criterion = nn.CosineEmbeddingLoss()
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(epochs):
        total_loss = 0.0
        for images, captions in dataloader:
            images = images.to(device)
            tokens = clip.tokenize(captions).to(device)

            with torch.no_grad():
                image_features = model.encode_image(images)
                text_features = model.encode_text(tokens)

            preds = head(image_features)
            targets = torch.ones(preds.shape[0]).to(device)
            loss = criterion(preds, text_features, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"📘 Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        writer.add_scalar("Train/Loss", avg_loss, epoch)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(head.state_dict(), save_path)
    print(f"✅ Saved linear head to {save_path}")


if __name__ == "__main__":
    train_clip_linear()
