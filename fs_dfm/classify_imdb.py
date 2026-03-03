#
# IMDB Classification Inference with Fine-tuned FS-DFM
#
# Two inference methods:
#   1. Single forward pass (fast): compare logits at the label position
#   2. Iterative sampling: use sample_masked() to generate the label token
#
# Usage:
#   python classify_imdb.py \
#       --checkpoint_path ./imdb_finetuned/best_checkpoint.pth \
#       --method logits \
#       --block_size 512
#

import argparse
import os
import sys
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import Transformer
from logic.flow import get_source_distribution, get_path
from logic.state import WrappedModel
from data.imdb_data import IMDBClassificationDataset, SENTIMENT_PREFIX, LABEL_TEXT

from flow_matching.solver.discrete_solver import (
    MixtureDiscreteEulerSolver,
)


def load_model(checkpoint_path, model_config, vocab_size, device):
    """Load the fine-tuned teacher model."""
    model = Transformer(
        config=model_config,
        vocab_size=vocab_size,
        masked=False,
        dt_conditioned=False,
    ).to(device)

    loaded = torch.load(checkpoint_path, map_location=device, weights_only=True)
    for key in ["teacher_model"]:
        if key in loaded:
            model.load_state_dict(loaded[key], strict=False)
            print(f"Loaded model from key '{key}'")
            break

    model.eval()
    return model


@torch.no_grad()
def classify_with_logits(
    model: Transformer,
    input_ids: torch.Tensor,
    label_token_pos: torch.Tensor,
    pos_token_id: int,
    neg_token_id: int,
    time_value: float = 0.95,
    source_distribution=None,
) -> torch.Tensor:
    """
    Single forward pass classification.

    The sequence has MASK at the label position and clean tokens elsewhere.
    We run the model at a high time value (near t=1.0, meaning mostly clean)
    and read the logits at the label position to compare P(positive) vs P(negative).

    Args:
        model: Fine-tuned FS-DFM teacher model.
        input_ids: [B, L] token sequences with MASK at label position.
        label_token_pos: [B] position index of the label token.
        pos_token_id: Token ID for " positive".
        neg_token_id: Token ID for " negative".
        time_value: Diffusion time (0=noise, 1=clean). Use ~0.95 since
                     most tokens are clean but the label position is masked.

    Returns:
        [B] tensor of predicted labels (0=negative, 1=positive).
    """
    B = input_ids.shape[0]
    device = input_ids.device

    time = torch.full((B,), time_value, device=device)
    x_0 = source_distribution.sample_like(input_ids)
    x_0 = torch.where(
        torch.arange(x_0.shape[1], device=x_0.device)[None, :] < label_token_pos[:, None],
        input_ids,
        x_0
    )
    logits = model(x_t=x_0, time=time)  # [B, L, vocab_size+1]

    # Extract logits at label positions
    predictions = []
    for i in range(B):
        pos = label_token_pos[i].item()
        pos_logit = logits[i, pos, pos_token_id]
        neg_logit = logits[i, pos, neg_token_id]
        predictions.append(1 if pos_logit > neg_logit else 0)

    return torch.tensor(predictions, device=device)


@torch.no_grad()
def classify_with_sampling(
    model: Transformer,
    input_ids: torch.Tensor,
    label_token_pos: torch.Tensor,
    pos_token_id: int,
    neg_token_id: int,
    vocab_size: int,
    path,
    sampling_steps: int = 64,
) -> torch.Tensor:
    """
    Iterative sampling classification using sample_masked().

    Starts from the input sequence (review=clean, label=MASK), runs the
    diffusion sampling process, and reads the generated token at the label position.

    Args:
        model: Fine-tuned FS-DFM teacher model.
        input_ids: [B, L] token sequences with MASK at label position.
        label_token_pos: [B] position index of the label token.
        pos_token_id: Token ID for " positive".
        neg_token_id: Token ID for " negative".
        vocab_size: Base vocab size (mask token = vocab_size).
        path: MixtureDiscreteProbPath for the solver.
        sampling_steps: Number of denoising steps.

    Returns:
        [B] tensor of predicted labels (0=negative, 1=positive, -1=uncertain).
    """
    B, L = input_ids.shape
    device = input_ids.device

    # Build edit_mask: True only at label positions
    edit_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
    for i in range(B):
        edit_mask[i, label_token_pos[i].item()] = True

    # Wrap model to output probabilities (solver expects softmax output)
    wrapped_model = WrappedModel(model=model)

    solver = MixtureDiscreteEulerSolver(
        model=wrapped_model,
        path=path,
        vocabulary_size=vocab_size + 1,  # +1 for mask token
    )

    step_size = 1.0 / sampling_steps
    result = solver.sample_masked(
        x_init=input_ids,
        step_size=step_size,
        edit_mask=edit_mask,
        time_grid=torch.tensor([0.0, 1.0]),
    )

    # Read predictions
    predictions = []
    for i in range(B):
        pos = label_token_pos[i].item()
        generated_token = result[i, pos].item()
        if generated_token == pos_token_id:
            predictions.append(1)
        elif generated_token == neg_token_id:
            predictions.append(0)
        else:
            # Model generated an unexpected token — fall back to logit comparison
            predictions.append(-1)

    return torch.tensor(predictions, device=device)


def build_masked_sequences(
    texts: List[str],
    tokenizer: GPT2Tokenizer,
    vocab_size: int,
    block_size: int,
):
    """
    Build input sequences for inference with MASK at the label position.

    Returns:
        input_ids: [B, block_size] tensor
        label_token_pos: [B] tensor
    """
    mask_token = vocab_size  # mask token ID = vocab_size (convention for mask source)
    eos_id = tokenizer.eos_token_id
    prefix_ids = tokenizer.encode(SENTIMENT_PREFIX, add_special_tokens=False)
    overhead = len(prefix_ids) + 1 + 1  # prefix + mask_slot + EOS
    max_review_len = block_size - overhead

    all_seqs = []
    all_positions = []

    for text in texts:
        review_ids = tokenizer.encode(text, add_special_tokens=False)[:max_review_len]
        label_pos = len(review_ids) + len(prefix_ids)

        seq = review_ids + prefix_ids + [mask_token] + [eos_id]
        if len(seq) < block_size:
            seq = seq + [eos_id] * (block_size - len(seq))

        all_seqs.append(seq)
        all_positions.append(label_pos)

    return (
        torch.tensor(all_seqs, dtype=torch.long),
        torch.tensor(all_positions, dtype=torch.long),
    )


def evaluate_imdb(
    model,
    tokenizer,
    device,
    method="logits",
    block_size=512,
    batch_size=32,
    time_value=0.95,
    sampling_steps=64,
    max_samples=None,
    vocab_size=None,
    path=None,
    source_distribution=None,
    dataset_dir=None
):
    """Evaluate on the full IMDB test set."""
    vocab_size = vocab_size or tokenizer.vocab_size

    # Get label token IDs
    pos_token_id = tokenizer.encode(" positive", add_special_tokens=False)[0]
    neg_token_id = tokenizer.encode(" negative", add_special_tokens=False)[0]

    # Load test dataset
    test_ds = IMDBClassificationDataset(
        split="test",
        block_size=block_size,
        tokenizer=tokenizer,
        max_samples=max_samples,
        cache_dir=dataset_dir,
    )
    print(f"Evaluating on {len(test_ds)} test samples with method='{method}'")

    correct = 0
    total = 0
    uncertain = 0

    for i in range(0, len(test_ds), batch_size):
        batch_indices = range(i, min(i + batch_size, len(test_ds)))
        batch = [test_ds[j] for j in batch_indices]

        # Build masked sequences for inference
        texts = [test_ds.data[j]["text"] for j in batch_indices]
        labels = torch.tensor([test_ds.data[j]["label"] for j in batch_indices])

        input_ids, label_token_pos = build_masked_sequences(
            texts, tokenizer, vocab_size, block_size
        )
        input_ids = input_ids.to(device)
        label_token_pos = label_token_pos.to(device)

        if method == "logits":
            preds = classify_with_logits(
                model, input_ids, label_token_pos,
                pos_token_id, neg_token_id, time_value,
                source_distribution=source_distribution,
            )
        elif method == "sampling":
            preds = classify_with_sampling(
                model, input_ids, label_token_pos,
                pos_token_id, neg_token_id, vocab_size, path, sampling_steps,
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        preds = preds.cpu()
        for pred, label in zip(preds, labels):
            if pred.item() == -1:
                uncertain += 1
            elif pred.item() == label.item():
                correct += 1
            total += 1

        if (i // batch_size) % 10 == 0:
            acc = correct / total if total > 0 else 0
            print(f"  Progress: {total}/{len(test_ds)} | Accuracy: {acc:.4f}")

    accuracy = correct / total if total > 0 else 0
    print(f"\nFinal Accuracy: {accuracy:.4f} ({correct}/{total})")
    if uncertain > 0:
        print(f"Uncertain predictions (unexpected tokens): {uncertain}")

    return accuracy


def main():
    parser = argparse.ArgumentParser(description="IMDB Classification with FS-DFM")

    parser.add_argument("--checkpoint_path", default="/vepfs/jinke/bw-data/ml-fs-dfm/outputs/imdb_ft_ddp/checkpoint_step_8000.pth", type=str)
    parser.add_argument("--method", type=str, default="logits",
                        choices=["logits", "sampling"])
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--time_value", type=float, default=0.95,
                        help="Diffusion time for logits method (0.9-0.99)")
    parser.add_argument("--sampling_steps", type=int, default=10,
                        help="Number of denoising steps for sampling method")
    parser.add_argument("--max_samples", type=int, default=1000,
                        help="Max test samples (None=all 25k)")

    # Model config
    parser.add_argument("--hidden_size", type=int, default=2048)
    parser.add_argument("--cond_dim", type=int, default=256)
    parser.add_argument("--n_blocks", type=int, default=21)
    parser.add_argument("--n_heads", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2Tokenizer.from_pretrained("/vepfs/jinke/bw-data/ml-fs-dfm/models/gpt2")
    vocab_size = tokenizer.vocab_size

    model_config = {
        "hidden_size": args.hidden_size,
        "cond_dim": args.cond_dim,
        "length": args.block_size,
        "n_blocks": args.n_blocks,
        "n_heads": args.n_heads,
        "dropout": args.dropout,
    }

    model = load_model(args.checkpoint_path, model_config, vocab_size, device)

    path = get_path(scheduler_type="polynomial", exponent=1.0)
    source_distribution = get_source_distribution("uniform", vocab_size)

    evaluate_imdb(
        model=model,
        tokenizer=tokenizer,
        device=device,
        method=args.method,
        block_size=args.block_size,
        batch_size=args.batch_size,
        time_value=args.time_value,
        sampling_steps=args.sampling_steps,
        max_samples=args.max_samples,
        vocab_size=vocab_size,
        path=path,
        source_distribution=source_distribution,
        dataset_dir="/vepfs/jinke/bw-data/ml-fs-dfm/datasets/imdb",
    )


if __name__ == "__main__":
    main()
