import argparse
import torch
from train import train
from train_clip_linear import train_clip_linear
from train_clip_full import train as train_clip_full
from eval_recall import evaluate as eval_recall
from evaluate_caption import evaluate_from_model
from models.clip_wrapper import CLIPWrapper
from multiview_dataset import MultiViewDataset
from torch.utils.data import DataLoader
from utils import load_clip_preprocessors


def run_train(args):
    if args.mode == "linear":
        train_clip_linear(
            jsonl_path=args.jsonl,
            image_dir=args.images,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            save_path=args.save_path,
            log_dir=args.log_dir
        )
    elif args.mode == "full":
        train_clip_full(
            jsonl_path=args.jsonl,
            image_dir=args.images,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            fine_tune="full",
            save_path=args.save_path,
            log_dir=args.log_dir
        )
    else:
        train(
            jsonl_path=args.jsonl,
            image_dir=args.images,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            fine_tune="linear",
            save_path=args.save_path,
            log_dir=args.log_dir
        )


def run_eval_recall(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model, preprocess, tokenizer = load_clip_preprocessors()
    model = CLIPWrapper(base_model, train_mode="eval")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()

    dataset = MultiViewDataset(args.jsonl, args.images, preprocess)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    recall_scores = eval_recall(model, dataloader, device=device)
    print(" Recall@K:", recall_scores)


def run_eval_caption(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model, preprocess, tokenizer = load_clip_preprocessors()
    model = CLIPWrapper(base_model, train_mode="eval")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)

    dataset = MultiViewDataset(args.jsonl, args.images, preprocess)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    scores = evaluate_from_model(model, dataloader, device=device)
    print(" Caption Evaluation:", scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", help="Sub-command")

    # Train command
    parser_train = subparsers.add_parser("train", help="Train CLIP")
    parser_train.add_argument("--mode", choices=["base", "linear", "full"], default="base")
    parser_train.add_argument("--jsonl", default="./data/pix3d/usable_samples.jsonl")
    parser_train.add_argument("--images", default="./data/pix3d/renders")
    parser_train.add_argument("--batch_size", type=int, default=32)
    parser_train.add_argument("--epochs", type=int, default=10)
    parser_train.add_argument("--lr", type=float, default=1e-5)
    parser_train.add_argument("--save_path", default="./weights/clip.pt")
    parser_train.add_argument("--log_dir", default="./logs")

    # Recall eval
    parser_recall = subparsers.add_parser("eval_recall", help="Evaluate Recall@K")
    parser_recall.add_argument("--model_path", required=True)
    parser_recall.add_argument("--jsonl", default="./data/pix3d/usable_samples.jsonl")
    parser_recall.add_argument("--images", default="./data/pix3d/renders")
    parser_recall.add_argument("--batch_size", type=int, default=32)

    # Caption eval
    parser_caption = subparsers.add_parser("eval_caption", help="Evaluate BLEU/METEOR")
    parser_caption.add_argument("--model_path", required=True)
    parser_caption.add_argument("--jsonl", default="./data/pix3d/usable_samples.jsonl")
    parser_caption.add_argument("--images", default="./data/pix3d/renders")
    parser_caption.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()

    if args.command == "train":
        run_train(args)
    elif args.command == "eval_recall":
        run_eval_recall(args)
    elif args.command == "eval_caption":
        run_eval_caption(args)
    else:
        print(" Please specify a valid command: train / eval_recall / eval_caption")
