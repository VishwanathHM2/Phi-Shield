"""
models/bilstm_crf_model.py

Production-grade BiLSTM + CRF sequence labeling model for PHI NER.

Architecture:
  Embedding → BiLSTM → Dropout → Linear → CRF (Viterbi decoding)

Loss: CRF negative log-likelihood
Decoding: Viterbi algorithm
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple


# ─── CRF Layer ────────────────────────────────────────────────────────────────

class CRF(nn.Module):
    """
    Conditional Random Field layer.

    Implements:
    - Forward algorithm for log-likelihood (training)
    - Viterbi algorithm for decoding (inference)
    - Constrained transitions for BIO tagging
    """

    def __init__(self, num_tags: int, pad_idx: int = 0):
        super().__init__()
        self.num_tags = num_tags
        self.pad_idx  = pad_idx

        # Transition matrix: transitions[i][j] = score of transitioning FROM tag j TO tag i
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        nn.init.uniform_(self.transitions, -0.1, 0.1)

        # Start and end transition scores
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions   = nn.Parameter(torch.empty(num_tags))
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions,   -0.1, 0.1)

    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute negative log-likelihood loss.

        Args:
            emissions: (batch, seq_len, num_tags) — BiLSTM output logits
            tags:      (batch, seq_len)           — gold tag indices
            mask:      (batch, seq_len) bool       — True for real tokens

        Returns:
            Scalar mean NLL loss.
        """
        log_likelihood = self._compute_log_likelihood(emissions, tags, mask)
        return -log_likelihood.mean()

    def decode(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
    ) -> List[List[int]]:
        """
        Viterbi decoding.

        Args:
            emissions: (batch, seq_len, num_tags)
            mask:      (batch, seq_len) bool

        Returns:
            List of predicted tag index sequences (one per batch item, unpadded).
        """
        return self._viterbi_decode(emissions, mask)

    # ─── Private: Forward algorithm ──────────────────────────────────────────

    def _compute_log_likelihood(self, emissions, tags, mask):
        batch_size, seq_len, _ = emissions.shape

        # Score of the gold sequence
        gold_score = self._compute_score(emissions, tags, mask)

        # Log partition function (normaliser)
        log_Z = self._compute_log_partition(emissions, mask)

        return gold_score - log_Z

    def _compute_score(self, emissions, tags, mask):
        """Sum of emission + transition scores for the gold path."""
        batch_size, seq_len, _ = emissions.shape

        # Start transition + first emission
        score = self.start_transitions[tags[:, 0]]
        score += emissions[:, 0, :].gather(1, tags[:, 0].unsqueeze(1)).squeeze(1)

        for t in range(1, seq_len):
            active = mask[:, t]
            # Transition from previous tag to current tag
            trans = self.transitions[tags[:, t], tags[:, t - 1]]
            emit  = emissions[:, t, :].gather(1, tags[:, t].unsqueeze(1)).squeeze(1)
            score += torch.where(active, trans + emit, torch.zeros_like(score))

        # End transition
        # Find last real tag for each sentence
        last_tag = tags.gather(
            1, (mask.long().sum(1) - 1).clamp(min=0).unsqueeze(1)
        ).squeeze(1)
        score += self.end_transitions[last_tag]
        return score

    def _compute_log_partition(self, emissions, mask):
        """Log-sum-exp forward pass."""
        batch_size, seq_len, num_tags = emissions.shape

        # (batch, num_tags) — alpha at t=0
        alpha = self.start_transitions + emissions[:, 0, :]

        for t in range(1, seq_len):
            # (batch, 1, num_tags) broadcast
            alpha_t = alpha.unsqueeze(2)
            # (1, num_tags, num_tags) transitions
            trans_t = self.transitions.unsqueeze(0)
            # (batch, num_tags)
            scores = torch.logsumexp(alpha_t + trans_t, dim=1) + emissions[:, t, :]

            active = mask[:, t].unsqueeze(1).expand_as(scores)
            alpha = torch.where(active, scores, alpha)

        return torch.logsumexp(alpha + self.end_transitions, dim=1)

    # ─── Private: Viterbi ────────────────────────────────────────────────────

    def _viterbi_decode(self, emissions, mask):
        batch_size, seq_len, num_tags = emissions.shape

        viterbi  = self.start_transitions + emissions[:, 0, :]  # (batch, num_tags)
        backpointers = []

        for t in range(1, seq_len):
            # (batch, num_tags, 1)
            viterbi_t = viterbi.unsqueeze(2)
            # (1, num_tags, num_tags)
            trans_t = self.transitions.unsqueeze(0)
            # (batch, num_tags, num_tags): score of coming from each prev tag to each cur tag
            scores = viterbi_t + trans_t

            best_scores, best_tags = scores.max(dim=1)  # (batch, num_tags) each
            backpointers.append(best_tags)

            best_scores += emissions[:, t, :]
            active = mask[:, t].unsqueeze(1).expand_as(best_scores)
            viterbi = torch.where(active, best_scores, viterbi)

        # End transitions
        viterbi += self.end_transitions

        # Backtrace
        seq_lens = mask.long().sum(1)  # (batch,)
        best_paths = []

        for b in range(batch_size):
            # Best last tag
            last_tag = int(viterbi[b].argmax())
            path = [last_tag]

            for bp in reversed(backpointers[: seq_lens[b] - 1]):
                last_tag = int(bp[b, last_tag])
                path.append(last_tag)

            path.reverse()
            best_paths.append(path)

        return best_paths


# ─── BiLSTM-CRF Model ─────────────────────────────────────────────────────────

class BiLSTMCRF(nn.Module):
    """
    BiLSTM-CRF for clinical PHI named-entity recognition.

    Layers:
        1. Embedding
        2. Bidirectional LSTM
        3. Dropout
        4. Linear projection → tag space
        5. CRF (NLL loss / Viterbi decode)
    """

    def __init__(
        self,
        vocab_size: int,
        num_tags: int,
        embedding_dim: int = 128,
        hidden_dim: int    = 256,
        num_layers: int    = 2,
        dropout: float     = 0.3,
        pad_idx: int       = 0,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim    = hidden_dim
        self.num_tags      = num_tags

        # 1. Embedding
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_idx
        )

        # 2. BiLSTM
        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 3. Dropout
        self.dropout = nn.Dropout(dropout)

        # 4. Linear projection
        self.linear = nn.Linear(hidden_dim, num_tags)

        # 5. CRF
        self.crf = CRF(num_tags, pad_idx=pad_idx)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        for name, param in self.bilstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def _get_emissions(self, word_ids: torch.Tensor) -> torch.Tensor:
        """Run embedding + BiLSTM + dropout + linear to get emission scores."""
        # (batch, seq_len, embedding_dim)
        embeds = self.dropout(self.embedding(word_ids))

        # (batch, seq_len, hidden_dim)
        lstm_out, _ = self.bilstm(embeds)
        lstm_out = self.dropout(lstm_out)

        # (batch, seq_len, num_tags)
        emissions = self.linear(lstm_out)
        return emissions

    def forward(
        self,
        word_ids: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Training forward pass — returns CRF NLL loss.

        Args:
            word_ids: (batch, seq_len) int tensor
            tags:     (batch, seq_len) int tensor
            mask:     (batch, seq_len) bool tensor

        Returns:
            Scalar loss.
        """
        emissions = self._get_emissions(word_ids)
        loss = self.crf(emissions, tags, mask)
        return loss

    def predict(
        self,
        word_ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> List[List[int]]:
        """
        Inference: Viterbi decode.

        Args:
            word_ids: (batch, seq_len)
            mask:     (batch, seq_len) bool

        Returns:
            List of predicted tag index sequences.
        """
        emissions = self._get_emissions(word_ids)
        return self.crf.decode(emissions, mask)
