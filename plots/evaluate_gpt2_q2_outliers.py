#!/usr/bin/env python3

import argparse
import copy
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


MODEL_NAME = "gpt2"
KMEANS_ITERS = 50
TERNARY_GRID = 80
FIT_SAMPLE_SIZE = 250_000
SEED = 42


def iter_target_weights(model):
    for name, param in model.named_parameters():
        if not name.endswith(".weight"):
            continue

        if ".attn." in name:
            group = "Attention"
        elif ".mlp." in name:
            group = "MLP"
        else:
            continue

        if not torch.is_floating_point(param):
            continue

        yield group, name, param


def sample_values(x, max_n, generator):
    x = x.detach().cpu().float().reshape(-1)

    if x.numel() <= max_n:
        return x

    idx = torch.randint(
        0,
        x.numel(),
        (max_n,),
        generator=generator,
    )

    return x[idx]


def fit_ternary(x):
    abs_x = x.abs()
    max_abs = float(torch.quantile(abs_x, 0.9995).item())

    thresholds = torch.linspace(
        0.0,
        max_abs,
        TERNARY_GRID,
    )

    best = None

    for delta in thresholds:
        mask = abs_x > delta

        if not mask.any():
            continue

        alpha = abs_x[mask].mean()

        q = torch.zeros_like(x)
        q[mask] = torch.sign(x[mask]) * alpha

        mse = torch.mean((x - q) ** 2).item()

        if best is None or mse < best["mse"]:
            best = {
                "delta": float(delta.item()),
                "alpha": float(alpha.item()),
                "mse": mse,
            }

    return best


def quantize_ternary(x, params):
    delta = params["delta"]
    alpha = params["alpha"]

    q = torch.zeros_like(x)
    mask = x.abs() > delta
    q[mask] = torch.sign(x[mask]) * alpha

    return q


def fit_q2_optimized(x):
    """
    Symmetric four-level quantizer:
        {-alpha, -beta, +beta, +alpha}

    Fit alpha/beta by 1-D k-means over |w|.
    """
    values = x.abs()

    c1 = torch.quantile(values, 0.35)
    c2 = torch.quantile(values, 0.85)

    for _ in range(KMEANS_ITERS):
        d1 = (values - c1).abs()
        d2 = (values - c2).abs()

        cluster1 = d1 <= d2
        cluster2 = ~cluster1

        new_c1 = values[cluster1].mean() if cluster1.any() else c1
        new_c2 = values[cluster2].mean() if cluster2.any() else c2

        if (
            torch.isclose(new_c1, c1, atol=1e-8)
            and torch.isclose(new_c2, c2, atol=1e-8)
        ):
            c1 = new_c1
            c2 = new_c2
            break

        c1 = new_c1
        c2 = new_c2

    beta = float(min(c1.item(), c2.item()))
    alpha = float(max(c1.item(), c2.item()))

    return {
        "beta": beta,
        "alpha": alpha,
        "threshold": (alpha + beta) / 2.0,
    }


def quantize_q2_optimized(x, params):
    alpha = params["alpha"]
    beta = params["beta"]
    threshold = params["threshold"]

    magnitude = torch.where(
        x.abs() <= threshold,
        torch.full_like(x, beta),
        torch.full_like(x, alpha),
    )

    return torch.sign(x) * magnitude


def tensor_metrics(original, reconstructed):
    a = original.detach().cpu().float().reshape(-1)
    b = reconstructed.detach().cpu().float().reshape(-1)

    error = a - b
    mse = torch.mean(error ** 2).item()

    denom = (
        torch.linalg.vector_norm(a)
        * torch.linalg.vector_norm(b)
    )

    cosine = (
        torch.dot(a, b) / denom
    ).item() if denom > 0 else float("nan")

    return {
        "mse": mse,
        "cosine": cosine,
        "max_error": error.abs().max().item(),
    }


def quantize_model(
    source_model,
    scheme,
    outlier_threshold=0.5,
):
    model = copy.deepcopy(source_model)

    source_params = dict(source_model.named_parameters())
    target_params = dict(model.named_parameters())

    generator = torch.Generator().manual_seed(SEED)

    rows = []

    total_weights = 0
    total_outliers = 0
    weighted_mse_sum = 0.0

    dot_ab = 0.0
    norm_a2 = 0.0
    norm_b2 = 0.0

    with torch.no_grad():
        for group, name, source_param in iter_target_weights(source_model):
            original = source_param.detach().cpu().float()
            fit_values = sample_values(
                original,
                FIT_SAMPLE_SIZE,
                generator,
            )

            if scheme == "ternary":
                params = fit_ternary(fit_values)
                reconstructed = quantize_ternary(
                    original,
                    params,
                )
                outlier_mask = torch.zeros_like(
                    original,
                    dtype=torch.bool,
                )

            elif scheme in {"q2", "q2_outliers"}:
                params = fit_q2_optimized(fit_values)
                reconstructed = quantize_q2_optimized(
                    original,
                    params,
                )

                if scheme == "q2_outliers":
                    outlier_mask = original.abs() > outlier_threshold

                    # Sparse FP16 residual correction.
                    # This simulates:
                    #   W ~= W_q2 + sparse_fp16_residual
                    residual_fp16 = (
                        original[outlier_mask]
                        - reconstructed[outlier_mask]
                    ).half()

                    reconstructed[outlier_mask] += (
                        residual_fp16.float()
                    )
                else:
                    outlier_mask = torch.zeros_like(
                        original,
                        dtype=torch.bool,
                    )

            else:
                raise ValueError(f"Unknown scheme: {scheme}")

            target_params[name].copy_(
                reconstructed.to(
                    dtype=target_params[name].dtype,
                    device=target_params[name].device,
                )
            )

            m = tensor_metrics(
                original,
                reconstructed,
            )

            n = original.numel()
            n_outliers = int(outlier_mask.sum().item())

            total_weights += n
            total_outliers += n_outliers
            weighted_mse_sum += m["mse"] * n

            a = original.reshape(-1).double()
            b = reconstructed.reshape(-1).double()

            dot_ab += torch.dot(a, b).item()
            norm_a2 += torch.dot(a, a).item()
            norm_b2 += torch.dot(b, b).item()

            rows.append(
                {
                    "group": group,
                    "tensor": name,
                    "n_weights": n,
                    "mse": m["mse"],
                    "cosine": m["cosine"],
                    "max_error": m["max_error"],
                    "outliers": n_outliers,
                    "outlier_fraction": n_outliers / n,
                    **{
                        k: v
                        for k, v in params.items()
                        if k != "mse"
                    },
                }
            )

    global_cosine = (
        dot_ab / math.sqrt(norm_a2 * norm_b2)
        if norm_a2 > 0 and norm_b2 > 0
        else float("nan")
    )

    stats = {
        "scheme": scheme,
        "n_quantized_weights": total_weights,
        "outliers": total_outliers,
        "outlier_fraction": total_outliers / total_weights,
        "weight_mse": weighted_mse_sum / total_weights,
        "weight_cosine": global_cosine,
    }

    return model, pd.DataFrame(rows), stats


def load_eval_text(args):
    if args.text_file:
        return Path(args.text_file).read_text(
            encoding="utf-8",
        )

    if args.text:
        return args.text

    # A small built-in corpus for a smoke test.
    # For publication-quality perplexity, pass a real text corpus
    # through --text-file.
    return (
        "Machine learning models represent information through large "
        "collections of numerical parameters. Transformer language models "
        "use attention mechanisms and feed-forward neural networks to "
        "predict the next token in a sequence. Quantization reduces the "
        "number of bits required to store model weights and can improve "
        "memory bandwidth efficiency during inference. "
    ) * 50


@torch.inference_mode()
def perplexity(
    model,
    tokenizer,
    text,
    device,
    max_length=1024,
    stride=512,
):
    encodings = tokenizer(
        text,
        return_tensors="pt",
    )

    input_ids = encodings.input_ids

    if input_ids.shape[1] < 2:
        raise ValueError(
            "Evaluation text is too short for perplexity."
        )

    model = model.to(device)
    model.eval()

    seq_len = input_ids.size(1)

    nll_sum = 0.0
    n_tokens = 0

    prev_end_loc = 0

    for begin_loc in range(
        0,
        seq_len,
        stride,
    ):
        end_loc = min(
            begin_loc + max_length,
            seq_len,
        )

        trg_len = end_loc - prev_end_loc

        input_chunk = input_ids[
            :,
            begin_loc:end_loc,
        ].to(device)

        target_ids = input_chunk.clone()

        if trg_len < input_chunk.size(1):
            target_ids[
                :,
                :-trg_len,
            ] = -100

        outputs = model(
            input_chunk,
            labels=target_ids,
        )

        valid_tokens = (
            target_ids[:, 1:] != -100
        ).sum().item()

        nll_sum += (
            outputs.loss.item()
            * valid_tokens
        )

        n_tokens += valid_tokens

        prev_end_loc = end_loc

        if end_loc == seq_len:
            break

    return math.exp(
        nll_sum / n_tokens
    )


def estimate_size_bytes(
    stats,
    scheme,
    index_bytes=4,
):
    n = stats["n_quantized_weights"]

    # Packed 2-bit dense matrix.
    dense_q2_bytes = math.ceil(n * 2 / 8)

    # Two FP16 levels per tensor are tiny compared with weights;
    # ignore here for a simple lower-order estimate.

    if scheme != "q2_outliers":
        return dense_q2_bytes

    # Sparse residual:
    #   int32 flat index + FP16 residual
    per_outlier = index_bytes + 2

    return (
        dense_q2_bytes
        + stats["outliers"] * per_outlier
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default=MODEL_NAME,
    )

    parser.add_argument(
        "--text-file",
        default=None,
    )

    parser.add_argument(
        "--text",
        default=None,
    )

    parser.add_argument(
        "--device",
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    parser.add_argument(
        "--outlier-threshold",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=512,
    )

    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    print(f"Device: {args.device}")

    tokenizer = GPT2TokenizerFast.from_pretrained(
        args.model
    )

    baseline_model = GPT2LMHeadModel.from_pretrained(
        args.model
    )

    baseline_model.eval()

    text = load_eval_text(args)

    token_count = tokenizer(
        text,
        return_tensors="pt",
    ).input_ids.shape[1]

    print(f"Evaluation tokens: {token_count:,}")

    results = []

    print("\n[1/4] Baseline FP32")
    baseline_ppl = perplexity(
        baseline_model,
        tokenizer,
        text,
        args.device,
        args.max_length,
        args.stride,
    )

    target_n = sum(
        p.numel()
        for _, _, p in iter_target_weights(
            baseline_model
        )
    )

    baseline_bytes = target_n * 4

    results.append(
        {
            "scheme": "fp32_baseline",
            "perplexity": baseline_ppl,
            "ppl_delta": 0.0,
            "ppl_ratio": 1.0,
            "weight_mse": 0.0,
            "weight_cosine": 1.0,
            "outlier_fraction": 0.0,
            "estimated_bytes": baseline_bytes,
            "compression_vs_fp32": 1.0,
        }
    )

    schemes = [
        "ternary",
        "q2",
        "q2_outliers",
    ]

    per_tensor_tables = []

    for idx, scheme in enumerate(
        schemes,
        start=2,
    ):
        print(f"\n[{idx}/4] {scheme}")

        quant_model, tensor_df, stats = quantize_model(
            baseline_model,
            scheme,
            args.outlier_threshold,
        )

        ppl = perplexity(
            quant_model,
            tokenizer,
            text,
            args.device,
            args.max_length,
            args.stride,
        )

        estimated_bytes = estimate_size_bytes(
            stats,
            scheme,
        )

        results.append(
            {
                "scheme": scheme,
                "perplexity": ppl,
                "ppl_delta": ppl - baseline_ppl,
                "ppl_ratio": ppl / baseline_ppl,
                "weight_mse": stats["weight_mse"],
                "weight_cosine": stats["weight_cosine"],
                "outlier_fraction": stats["outlier_fraction"],
                "estimated_bytes": estimated_bytes,
                "compression_vs_fp32": (
                    baseline_bytes / estimated_bytes
                ),
            }
        )

        tensor_df.insert(
            0,
            "scheme",
            scheme,
        )

        per_tensor_tables.append(
            tensor_df
        )

        del quant_model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result_df = pd.DataFrame(results)

    print(
        "\n"
        "================ MODEL QUALITY ================\n"
    )

    display = result_df.copy()

    display["outlier_fraction_pct"] = (
        display["outlier_fraction"] * 100.0
    )

    display["estimated_MB"] = (
        display["estimated_bytes"]
        / 1024**2
    )

    cols = [
        "scheme",
        "perplexity",
        "ppl_delta",
        "ppl_ratio",
        "weight_mse",
        "weight_cosine",
        "outlier_fraction_pct",
        "estimated_MB",
        "compression_vs_fp32",
    ]

    print(
        display[cols]
        .round(6)
        .to_string(index=False)
    )

    result_df.to_csv(
        "gpt2_q2_perplexity_results.csv",
        index=False,
    )

    pd.concat(
        per_tensor_tables,
        ignore_index=True,
    ).to_csv(
        "gpt2_q2_per_tensor_results.csv",
        index=False,
    )

    print("\nSaved:")
    print("  gpt2_q2_perplexity_results.csv")
    print("  gpt2_q2_per_tensor_results.csv")


if __name__ == "__main__":
    main()
