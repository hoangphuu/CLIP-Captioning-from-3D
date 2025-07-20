CLIP-Captioning-from-3D/
├── data/
│   └── pix3d/
│       ├── img/                    # Raw RGB images
│       ├── mask/                   # Object segmentation masks
│       ├── model/                  # 3D mesh models (.obj)
│       ├── renders/                # Synthesized renders from 3D views (generated)
│       └── pix3d.json              # Metadata for each object-view pair
│       └── usable_samples.jsonl   # Preprocessed and filtered samples for training
│
├── models/
│   └── clip_wrapper.py            # Wrapper around CLIP for flexible training
│
├── scripts/
│   ├── render_from_obj.py         # Render .obj files to images (Multi-view)
│   └── generate_jsonl.py          # Extract usable samples into JSONL format
│
├── train.py                       # Main training script (linear or full)
├── train_clip_linear.py          # Linear-probe only (frozen CLIP)
├── train_clip_full.py            # Full fine-tuning
├── eval_recall.py                # Evaluate Recall@K (image ↔ text retrieval)
├── evaluate_caption.py           # Evaluate BLEU, METEOR (optional)
│
├── multiview_dataset.py          # Dataset class to load image-caption pairs
├── utils.py                      # Preprocessing utils (load CLIP, transforms, etc.)
│
├── weights/                      # Trained model checkpoints
├── logs/                         # TensorBoard logs
└── README.md


1. Preprocessing
   - Parse `pix3d.json` to filter valid samples
   - Render 3D models from multiple views using PyTorch3D (if needed)
   - Save metadata in `usable_samples.jsonl`

2. Dataset Preparation
   - `MultiViewDataset` loads (render, caption) pairs
   - Apply CLIP preprocessing to images and texts

3. Model Initialization
   - Load OpenAI CLIP (ViT-B/32 or ViT-B/16)
   - Wrap with `CLIPWrapper` to enable:
     - Linear probing (freeze CLIP, train projection)
     - Full fine-tuning (update entire backbone)

4. Training
   - Optimize cosine similarity (CosineEmbeddingLoss)
   - Log training loss via TensorBoard
   - Save model checkpoints (`.pt`)

5. Evaluation
   a. Retrieval (default):
      - Use `eval_recall.py`
      - Compute image2text and text2image Recall@1, 5, 10
   b. Caption quality (optional):
      - Use `evaluate_caption.py`
      - Compute BLEU, METEOR, ROUGE-L

6. Visualization (optional)
   - Plot Recall@K curves
   - Visualize attention weights (for interpretability)


Component	Mô tả
renders/	Ảnh được sinh từ 3D .obj (nhiều góc nhìn)
CLIPWrapper	Cho phép fine-tune linear hoặc full
MultiViewDataset	Load ảnh và caption dựa vào usable_samples.jsonl
CosineEmbeddingLoss	Loss function cho retrieval task
eval_recall.py	Đánh giá khả năng truy xuất (Recall@K)
evaluate_caption.py	Đánh giá chất lượng caption sinh ra (BLEU, METEOR) - tùy chọn

