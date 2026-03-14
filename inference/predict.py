"""
inference/predict.py

Inference pipeline for PHI detection using a trained BiLSTM-CRF model.
Provides token-level entity predictions with entity span reconstruction.
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import (
    BEST_MODEL_PATH,
    VOCAB_FILE,
    NUM_TAGS,
    EMBEDDING_DIM,
    HIDDEN_DIM,
    DROPOUT,
    PAD_IDX,
    MAX_SEQUENCE_LENGTH,
    IDX2TAG,
    UNK_IDX,
)

from utils.tokenizer import tokenize
from models.bilstm_crf_model import BiLSTMCRF
from dataset.vocab_builder import load_vocab


class PHIPredictor:
    """
    Wrapper for a trained BiLSTM-CRF model for PHI detection.

    Example:
        predictor = PHIPredictor()
        entities = predictor.predict_entities("Patient Ravi Kumar visited Apollo Hospital.")
    """

    def __init__(
        self,
        model_path: Path = BEST_MODEL_PATH,
        vocab_path: Path = VOCAB_FILE,
        device: Optional[torch.device] = None,
    ):

        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

        self.device = device

        if not vocab_path.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")

        self.word2idx = load_vocab(vocab_path)

        self.model = self._load_model(model_path)
        self.model.eval()

    def _load_model(self, model_path: Path) -> BiLSTMCRF:

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {model_path}\n"
                "Run training/train.py first."
            )

        checkpoint = torch.load(model_path, map_location=self.device)

        model = BiLSTMCRF(
            vocab_size=checkpoint.get("vocab_size", len(self.word2idx)),
            num_tags=checkpoint.get("num_tags", NUM_TAGS),
            embedding_dim=checkpoint.get("embedding_dim", EMBEDDING_DIM),
            hidden_dim=checkpoint.get("hidden_dim", HIDDEN_DIM),
            dropout=DROPOUT,
            pad_idx=PAD_IDX,
        ).to(self.device)

        model.load_state_dict(checkpoint["model"])

        return model

    def _encode_tokens(self, tokens: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert tokens → tensors"""

        tokens = tokens[:MAX_SEQUENCE_LENGTH]

        ids = [self.word2idx.get(tok, UNK_IDX) for tok in tokens]

        mask = [True] * len(ids)

        word_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        mask_t = torch.tensor([mask], dtype=torch.bool, device=self.device)

        return word_ids, mask_t

    def predict_tags(self, text: str) -> Tuple[List[str], List[str]]:
        """
        Predict BIO tags for text.

        Returns
        -------
        tokens : List[str]
        tags   : List[str]
        """

        tokens = tokenize(text)

        if not tokens:
            return [], []

        word_ids, mask = self._encode_tokens(tokens)

        with torch.no_grad():

            pred_ids = self.model.predict(word_ids, mask)[0]

        tags = [IDX2TAG.get(idx, "O") for idx in pred_ids]

        tokens = tokens[:len(tags)]

        return tokens, tags

    def predict_entities(self, text: str) -> List[Dict]:
        """
        Convert BIO tags → entity spans
        """

        tokens, tags = self.predict_tags(text)

        entities = []

        i = 0

        while i < len(tags):

            tag = tags[i]

            if tag.startswith("B-"):

                entity_type = tag[2:]

                j = i + 1

                while j < len(tags) and tags[j] == f"I-{entity_type}":
                    j += 1

                span_tokens = tokens[i:j]

                entities.append(
                    {
                        "entity_type": entity_type,
                        "tokens": span_tokens,
                        "text": " ".join(span_tokens),
                        "start_token": i,
                        "end_token": j - 1,
                    }
                )

                i = j

            else:

                i += 1

        return entities