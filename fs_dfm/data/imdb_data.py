#
# IMDB Sentiment Classification Dataset for FS-DFM
#
# Formats IMDB reviews as:
#   "{review_text} Sentiment: positive<|endoftext|>"
# or "negative", for generative classification with a diffusion model.
#


from typing import Optional

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import GPT2TokenizerFast


# Template tokens appended after the review text
SENTIMENT_PREFIX = " Sentiment:"

# In GPT2: " positive" -> [3967], " negative" -> [4633] (single tokens)
LABEL_TEXT = {0: " negative", 1: " positive"}


class IMDBClassificationDataset(Dataset):
    """
    IMDB dataset formatted for generative classification with FS-DFM.

    Each sample is a token sequence:
        [review_tokens...] [sentiment_prefix_tokens] [label_token] [EOS] [PAD...]

    The model is trained to denoise the entire sequence (standard DFM loss),
    and at inference the label_token position is masked for conditional generation.
    """

    def __init__(
        self,
        split: str = "train",
        block_size: int = 1024,
        tokenizer: Optional[GPT2TokenizerFast] = None,
        cache_dir: Optional[str] = None,
        max_samples: Optional[int] = None,
    ):
        self.tokenizer = tokenizer or GPT2TokenizerFast.from_pretrained("gpt2")
        self.block_size = block_size

        # EOS token (used for padding too, matching existing FS-DFM convention)
        self.eos_id = self.tokenizer.eos_token_id

        # Tokenize the fixed prefix and label tokens
        self.prefix_ids = self.tokenizer.encode(
            SENTIMENT_PREFIX, add_special_tokens=False
        )

        self.label_token_ids = {}
        for label_idx, label_text in LABEL_TEXT.items():
            ids = self.tokenizer.encode(label_text, add_special_tokens=False)
            assert len(ids) == 1, (
                f"Expected single token for '{label_text}', got {ids}. "
                f"If your tokenizer differs, adjust LABEL_TEXT."
            )
            self.label_token_ids[label_idx] = ids[0]

        # overhead = prefix + label_token + EOS
        self.overhead = len(self.prefix_ids) + 1 + 1
        self.max_review_len = block_size - self.overhead

        # Load IMDB
        ds = load_dataset("imdb", split=split, cache_dir=cache_dir)
        if max_samples is not None:
            ds = ds.select(range(min(len(ds), max_samples)))
        self.data = ds

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        label = item["label"]  # 0=negative, 1=positive

        # Tokenize review and truncate
        review_ids = self.tokenizer.encode(
            item["text"], add_special_tokens=False
        )[: self.max_review_len]

        # Build sequence: review + prefix + label + EOS
        label_token = self.label_token_ids[label]
        seq = review_ids + self.prefix_ids + [label_token] + [self.eos_id]

        # Track the position of the label token (for inference masking)
        label_token_pos = len(review_ids) + len(self.prefix_ids)

        # Pad to block_size with EOS (matching existing FS-DFM convention)
        if len(seq) < self.block_size:
            seq = seq + [self.eos_id] * (self.block_size - len(seq))

        return {
            "input_ids": torch.tensor(seq, dtype=torch.long),
            "label_token_pos": torch.tensor(label_token_pos, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
        }


def get_imdb_datasets(
    tokenizer: Optional[GPT2TokenizerFast] = None,
    block_size: int = 1024,
    cache_dir: Optional[str] = None,
    max_train_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
):
    """Convenience function to get both train and test datasets."""
    train_ds = IMDBClassificationDataset(
        split="train",
        block_size=block_size,
        tokenizer=tokenizer,
        cache_dir=cache_dir,
        max_samples=max_train_samples,
    )
    test_ds = IMDBClassificationDataset(
        split="test",
        block_size=block_size,
        tokenizer=tokenizer,
        cache_dir=cache_dir,
        max_samples=max_test_samples,
    )
    return train_ds, test_ds
