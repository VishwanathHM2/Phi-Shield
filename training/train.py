"""
training/train.py — Training loop with LR scheduler and early stopping.
"""
import sys, time, json, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config.config import (
    LEARNING_RATE, EPOCHS, GRAD_CLIP, CHECKPOINTS_DIR, BEST_MODEL_PATH,
    TRAIN_FILE, DEV_FILE, BATCH_SIZE, LOG_INTERVAL,
    LR_PATIENCE, LR_FACTOR, EMBEDDING_DIM, HIDDEN_DIM, DROPOUT,
    NUM_TAGS, MIN_VOCAB_FREQ, TAGS, TAG2IDX,
)
from dataset.vocab_builder import build_vocab, save_vocab, save_tags, VOCAB_FILE, TAG_FILE
from dataset.dataset_loader import get_dataloader
from models.bilstm_crf_model import BiLSTMCRF
from training.evaluate import evaluate_model


def train_epoch(model, loader, optimizer, device, epoch):
    model.train()
    total_loss, steps = 0.0, 0
    t0 = time.time()
    for i, (words, tags, mask) in enumerate(loader):
        words, tags, mask = words.to(device), tags.to(device), mask.to(device)
        optimizer.zero_grad()
        loss = model(words, tags, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        total_loss += loss.item()
        steps += 1
        if (i + 1) % LOG_INTERVAL == 0:
            print(f"  Epoch {epoch} | Step {i+1}/{len(loader)} | "
                  f"Loss: {total_loss/steps:.4f} | {time.time()-t0:.1f}s")
    return total_loss / max(steps, 1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")

    print("[Train] Building vocabulary...")
    word2idx = build_vocab(TRAIN_FILE, min_freq=MIN_VOCAB_FREQ)
    save_vocab(word2idx, VOCAB_FILE)
    save_tags(TAG2IDX, TAG_FILE)
    vocab_size = len(word2idx)
    print(f"[Train] Vocab: {vocab_size:,} | Tags: {NUM_TAGS}")

    train_loader = get_dataloader(TRAIN_FILE, word2idx, shuffle=True,  batch_size=BATCH_SIZE)
    dev_loader   = get_dataloader(DEV_FILE,   word2idx, shuffle=False, batch_size=BATCH_SIZE)
    print(f"[Train] Batches — Train: {len(train_loader)} | Dev: {len(dev_loader)}")

    model = BiLSTMCRF(
        vocab_size=vocab_size,
        num_tags=NUM_TAGS,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
    ).to(device)
    print(f"[Train] Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=LR_FACTOR,
                                  patience=LR_PATIENCE)

    best_f1    = 0.0
    no_improve = 0
    history    = []
    EARLY_STOP = 6

    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'='*60}\nEpoch {epoch}/{EPOCHS} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        train_loss       = train_epoch(model, train_loader, optimizer, device, epoch)
        val_loss, metrics = evaluate_model(model, dev_loader, device)

        # metrics keys: entity names + "overall", each with precision/recall/f1/support
        overall = metrics.get("overall", {})
        f1 = overall.get("f1",        0.0)
        p  = overall.get("precision", 0.0)
        r  = overall.get("recall",    0.0)

        print(f"\n[Epoch {epoch}] TrainLoss: {train_loss:.4f} | "
              f"ValLoss: {val_loss:.4f} | "
              f"P: {p:.4f} | R: {r:.4f} | F1: {f1:.4f}")

        # Per-class breakdown (skip "overall")
        for cls, m in metrics.items():
            if cls == "overall": continue
            print(f"  {cls:12s}  P:{m['precision']:.3f}  R:{m['recall']:.3f}  F1:{m['f1']:.3f}")

        scheduler.step(f1)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "f1": f1, "precision": p, "recall": r})

        if f1 > best_f1:
            best_f1    = f1
            no_improve = 0
            torch.save({
                "epoch":         epoch,
                "f1":            f1,
                "model":         model.state_dict(),
                "optimizer":     optimizer.state_dict(),
                "vocab_size":    vocab_size,
                "word2idx":      word2idx,
                "num_tags":      NUM_TAGS,
                "embedding_dim": EMBEDDING_DIM,
                "hidden_dim":    HIDDEN_DIM,
            }, BEST_MODEL_PATH)
            print(f"  ✓ Best model saved (F1={f1:.4f}) → {BEST_MODEL_PATH}")
        else:
            no_improve += 1
            print(f"  No improvement for {no_improve} epoch(s).")
            if no_improve >= EARLY_STOP:
                print(f"[Train] Early stopping at epoch {epoch}.")
                break

    json.dump(history, open(CHECKPOINTS_DIR / "history.json", "w"), indent=2)
    print(f"\n[Train] Best Val F1: {best_f1:.4f}")
    print("[Train] Training complete.")


if __name__ == "__main__":
    main()