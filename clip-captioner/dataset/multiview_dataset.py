import os
import json
from torch.utils.data import Dataset
from PIL import Image

class MultiViewDataset(Dataset):
    def __init__(self, jsonl_path, image_root, preprocess_fn):
        """
        Dataset cho ảnh render + caption từ Pix3D.

        Args:
            jsonl_path (str): Đường dẫn đến file .jsonl chứa {image_path, caption, model}
            image_root (str): Thư mục gốc chứa ảnh render
            preprocess_fn (Callable): Hàm tiền xử lý ảnh từ CLIP
        """
        self.samples = []
        self.image_root = image_root
        self.preprocess_fn = preprocess_fn

        with open(jsonl_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                image_path = os.path.join(self.image_root, os.path.basename(item["image_path"]))
                if os.path.exists(image_path):
                    self.samples.append({
                        "image_path": image_path,
                        "caption": item["caption"],
                        "model": item.get("model", None)
                    })

        assert len(self.samples) > 0, f"No valid samples found in {jsonl_path}"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = Image.open(sample["image_path"]).convert("RGB")
        image_tensor = self.preprocess_fn(image)  # Returns a tensor (3, 224, 224)

        caption = sample["caption"]

        return image_tensor, caption
