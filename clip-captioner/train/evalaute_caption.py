import evaluate
from tqdm import tqdm


def evaluate_caption(gold_captions, pred_captions):
    """
    gold_captions: List[str]
    pred_captions: List[str]
    """
    bleu = evaluate.load("bleu")
    meteor = evaluate.load("meteor")

    bleu_score = bleu.compute(predictions=pred_captions, references=[[g] for g in gold_captions])
    meteor_score = meteor.compute(predictions=pred_captions, references=gold_captions)

    return {
        "BLEU": bleu_score["bleu"],
        "METEOR": meteor_score["meteor"]
    }


def evaluate_from_model(model, dataloader, device="cuda"):
    model.eval()
    gold, pred = [], []
    with torch.no_grad():
        for images, captions in tqdm(dataloader):
            images = images.to(device)
            predictions = model.generate_caption(images)
            pred.extend(predictions)
            gold.extend(captions)

    return evaluate_caption(gold, pred)
