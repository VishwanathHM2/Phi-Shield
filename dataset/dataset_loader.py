"""
dataset/dataset_loader.py

PyTorch Dataset and DataLoader for BIO-tagged NER data.
Handles tokenization, padding, masking, and batch creation.
"""

import sys
import torch
from pathlib import Path
from typing import Dict, List, Tuple
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import (
    MAX_SEQUENCE_LENGTH,
    PAD_IDX,
    UNK_IDX,
    UNK_TOKEN,
    TAG2IDX,
    BATCH_SIZE,
)
from dataset.vocab_builder import load_conll


class PHIDataset(Dataset):
    """
    Dataset for PHI NER.
    Each item: (word_ids tensor, tag_ids tensor, mask tensor)
    """

    def __init__(
        self,
        filepath: Path,
        word2idx: Dict[str, int],
        tag2idx: Dict[str, int] = None,
        max_len: int = MAX_SEQUENCE_LENGTH,
    ):
        self.word2idx = word2idx
        self.tag2idx  = tag2idx or TAG2IDX
        self.max_len  = max_len

        raw = load_conll(filepath)
        self.samples = self._encode(raw)

    def _encode(self, raw_sentences):
        samples = []
        for tokens, tags in raw_sentences:
            # Truncate
            tokens = tokens[: self.max_len]
            tags   = tags[: self.max_len]

            word_ids = [
                self.word2idx.get(tok, UNK_IDX) for tok in tokens
            ]
            tag_ids = [
                self.tag2idx.get(tag, self.tag2idx.get("O", 0)) for tag in tags
            ]
            mask = [1] * len(word_ids)

            samples.append((word_ids, tag_ids, mask))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        word_ids, tag_ids, mask = self.samples[idx]
        return (
            torch.tensor(word_ids, dtype=torch.long),
            torch.tensor(tag_ids,  dtype=torch.long),
            torch.tensor(mask,     dtype=torch.bool),
        )


def collate_fn(batch):
    """
    Pad sequences in a batch to the same length.
    Returns (word_ids, tag_ids, mask) tensors of shape (batch, max_len).
    """
    word_seqs, tag_seqs, mask_seqs = zip(*batch)

    word_padded = pad_sequence(word_seqs, batch_first=True, padding_value=PAD_IDX)
    tag_padded  = pad_sequence(tag_seqs,  batch_first=True, padding_value=PAD_IDX)
    mask_padded = pad_sequence(mask_seqs, batch_first=True, padding_value=False)

    return word_padded, tag_padded, mask_padded


def get_dataloader(
    filepath: Path,
    word2idx: Dict[str, int],
    tag2idx: Dict[str, int] = None,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    dataset = PHIDataset(filepath, word2idx, tag2idx)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
