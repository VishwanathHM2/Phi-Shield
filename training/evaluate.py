"""
training/evaluate.py

Entity-level evaluation of BiLSTM-CRF model on a dataset split.
Returns per-entity and overall Precision / Recall / F1.
"""

import sys
from pathlib import Path
from typing import Tuple, Dict

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import IDX2TAG, BATCH_SIZE, TEST_FILE, BEST_MODEL_PATH, VOCAB_FILE
from utils.metrics import compute_entity_metrics, format_metrics_report


def evaluate_model(model, loader, device) -> Tuple[float, Dict]:
    """
    Run model on all batches in loader, compute entity-level metrics.

    Returns:
        (mean_loss, metrics_dict)
    """
    model.eval()
    total_loss   = 0.0
    n_batches    = 0
    all_true     = []
    all_pred     = []

    with torch.no_grad():
        for word_ids, tag_ids, mask in loader:
            word_ids = word_ids.to(device)
            tag_ids  = tag_ids.to(device)
            mask     = mask.to(device)

            loss     = model(word_ids, tag_ids, mask)
            pred_ids = model.predict(word_ids, mask)

            total_loss += loss.item()
            n_batches  += 1

            # Convert to tag strings for metric computation
            seq_lens = mask.long().sum(dim=1).cpu().tolist()
            for i, length in enumerate(seq_lens):
                true_tags = [IDX2TAG.get(t.item(), "O") for t in tag_ids[i, :length]]
                pred_tags = [IDX2TAG.get(t, "O") for t in pred_ids[i]]
                all_true.append(true_tags)
                all_pred.append(pred_tags)

    mean_loss = total_loss / max(n_batches, 1)
    metrics   = compute_entity_metrics(all_true, all_pred)
    return mean_loss, metrics


def main():
    """Standalone evaluation on test split."""
    from dataset.vocab_builder import load_vocab
    from dataset.dataset_loader import get_dataloader
    from models.bilstm_crf_model import BiLSTMCRF
    from config.config import (
        EMBEDDING_DIM, HIDDEN_DIM, DROPOUT, NUM_TAGS, PAD_IDX,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not BEST_MODEL_PATH.exists():
        print(f"No checkpoint found at {BEST_MODEL_PATH}. Train the model first.")
        return

    word2idx = load_vocab(VOCAB_FILE)
    vocab_size = len(word2idx)

    model = BiLSTMCRF(
        vocab_size=vocab_size,
        num_tags=NUM_TAGS,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
        pad_idx=PAD_IDX,
    ).to(device)

    ckpt = torch.load(BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded model from epoch {ckpt['epoch']} (best F1: {ckpt['best_f1']:.4f})")

    test_loader = get_dataloader(TEST_FILE, word2idx, shuffle=False, batch_size=BATCH_SIZE)
    loss, metrics = evaluate_model(model, test_loader, device)

    print(f"\nTest Loss: {loss:.4f}")
    print("\n" + format_metrics_report(metrics))


if __name__ == "__main__":
    main()
