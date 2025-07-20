import json
import os
from tqdm import tqdm

def generate_jsonl(
    pix3d_json_path="./data/pix3d/pix3d.json",
    output_path="./data/pix3d/usable_samples.jsonl",
    image_folder="./data/pix3d/img",
    render_folder="./data/pix3d/renders"
):
    with open(pix3d_json_path, "r") as f:
        data = json.load(f)

    with open(output_path, "w") as out_f:
        for item in tqdm(data, desc="Generating jsonl"):
            if not item.get("model") or not item.get("bbox") or not item.get("img"):
                continue

            image_path = os.path.join(image_folder, item["img"])
            render_path = os.path.join(render_folder, item["model"].replace(".obj", ".png"))
            caption = item.get("category", "object")

            if not os.path.exists(image_path) or not os.path.exists(render_path):
                continue

            json_obj = {
                "image_path": render_path,
                "caption": caption,
                "model": item["model"]
            }

            out_f.write(json.dumps(json_obj) + "\n")

    print(f"✅ Saved usable samples to {output_path}")

if __name__ == "__main__":
    generate_jsonl()
