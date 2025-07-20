import torch
import torch.nn.functional as F
from tqdm import tqdm


def compute_recall_at_k(image_features, text_features, ks=(1, 5, 10)):
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    sims = image_features @ text_features.T
    num_samples = sims.size(0)
    result = {}

    for k in ks:
        image2text_correct = sum(i in torch.topk(sims[i], k).indices for i in range(num_samples))
        text2image_correct = sum(i in torch.topk(sims[:, i], k).indices for i in range(num_samples))
        result[f"image2text_R@{k}"] = image2text_correct / num_samples
        result[f"text2image_R@{k}"] = text2image_correct / num_samples

    return result


def evaluate(model, dataloader, device="cuda"):
    model.eval()
    all_image_features, all_text_features = [], []
    with torch.no_grad():
        for images, captions in tqdm(dataloader, desc="Extracting embeddings"):
            images = images.to(device)
            tokens = model.tokenizer(captions).to(device)
            img_feat = model.encode_image(images)
            txt_feat = model.encode_text(tokens)
            all_image_features.append(img_feat)
            all_text_features.append(txt_feat)

    all_image_features = torch.cat(all_image_features, dim=0)
    all_text_features = torch.cat(all_text_features, dim=0)
    return compute_recall_at_k(all_image_features, all_text_features)
