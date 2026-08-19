#!/usr/bin/env python3

import math
import numpy as np
import pandas as pd
import torch
from transformers import GPT2LMHeadModel


MODEL_NAME = "gpt2"
FIT_SAMPLE_SIZE = 250_000
TERNARY_THRESHOLD_GRID = 80
UNIFORM_SCALE_GRID = 80
KMEANS_ITERS = 50
SEED = 42


def sample_values(x: torch.Tensor, max_n: int, generator: torch.Generator):
    x = x.reshape(-1)

    if x.numel() <= max_n:
        return x

    idx = torch.randint(
        0,
        x.numel(),
        (max_n,),
        generator=generator,
    )

    return x[idx]


def fit_ternary(x: torch.Tensor):
    """
    Fit symmetric ternary levels {-alpha, 0, +alpha}.

    We grid-search threshold delta.
    For each threshold:
      q = sign(x) for |x| > delta else 0
      alpha = mean(|x| on nonzero entries)
    and choose the threshold with minimum MSE.
    """
    abs_x = x.abs()

    max_abs = float(torch.quantile(abs_x, 0.9995).item())

    thresholds = torch.linspace(
        0.0,
        max_abs,
        TERNARY_THRESHOLD_GRID,
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
                "mse": mse,
                "delta": float(delta.item()),
                "alpha": float(alpha.item()),
            }

    return best


def quantize_ternary(x: torch.Tensor, params):
    delta = params["delta"]
    alpha = params["alpha"]

    mask = x.abs() > delta

    q = torch.zeros_like(x)
    q[mask] = torch.sign(x[mask]) * alpha

    return q


def fit_4level_symmetric(x: torch.Tensor):
    """
    Fit 4 symmetric levels:
        {-alpha, -beta, +beta, +alpha}

    Since the quantizer is symmetric, run 1D k-means on |x| with k=2.
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

    threshold = (alpha + beta) / 2.0

    return {
        "alpha": alpha,
        "beta": beta,
        "threshold": threshold,
    }


def quantize_4level_symmetric(x: torch.Tensor, params):
    alpha = params["alpha"]
    beta = params["beta"]
    threshold = params["threshold"]

    magnitude = torch.where(
        x.abs() <= threshold,
        torch.full_like(x, beta),
        torch.full_like(x, alpha),
    )

    return torch.sign(x) * magnitude


def fit_uniform_int2(x: torch.Tensor):
    """
    Symmetric 4-level uniform 2-bit quantizer:

        {-s, -s/3, +s/3, +s}

    Search s to minimize MSE.
    """
    abs_x = x.abs()

    p95 = float(torch.quantile(abs_x, 0.95).item())
    p9999 = float(torch.quantile(abs_x, 0.9999).item())

    low = max(p95, 1e-8)
    high = max(p9999, low * 1.01)

    scales = torch.linspace(
        low,
        high,
        UNIFORM_SCALE_GRID,
    )

    best = None

    for scale in scales:
        s = float(scale.item())

        levels = torch.tensor(
            [-s, -s / 3.0, s / 3.0, s],
            dtype=x.dtype,
        )

        distances = (x[:, None] - levels[None, :]).abs()
        indices = distances.argmin(dim=1)
        q = levels[indices]

        mse = torch.mean((x - q) ** 2).item()

        if best is None or mse < best["mse"]:
            best = {
                "mse": mse,
                "scale": s,
            }

    return best


def quantize_uniform_int2(x: torch.Tensor, params):
    s = params["scale"]

    levels = torch.tensor(
        [-s, -s / 3.0, s / 3.0, s],
        dtype=x.dtype,
    )

    distances = (x[:, None] - levels[None, :]).abs()
    indices = distances.argmin(dim=1)

    return levels[indices]


def metrics(original: torch.Tensor, quantized: torch.Tensor):
    error = original - quantized

    mse = torch.mean(error ** 2).item()
    rmse = math.sqrt(mse)
    mae = torch.mean(error.abs()).item()
    max_error = torch.max(error.abs()).item()

    denom = (
        torch.linalg.vector_norm(original)
        * torch.linalg.vector_norm(quantized)
    )

    cosine = (
        torch.dot(original, quantized) / denom
    ).item() if denom > 0 else float("nan")

    signal_power = torch.mean(original ** 2).item()

    sqnr_db = (
        10.0 * math.log10(signal_power / mse)
        if mse > 0 and signal_power > 0
        else float("inf")
    )

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "max_error": max_error,
        "cosine": cosine,
        "sqnr_db": sqnr_db,
    }


def select_weight_tensors(model):
    selected = []

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

        selected.append(
            (
                group,
                name,
                param.detach().cpu().float().reshape(-1),
            )
        )

    return selected


def main():
    print(f"Loading {MODEL_NAME}...")
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    model.eval()

    tensors = select_weight_tensors(model)

    generator = torch.Generator().manual_seed(SEED)

    rows = []

    for tensor_index, (group, name, weights) in enumerate(tensors, start=1):
        print(
            f"[{tensor_index:02d}/{len(tensors):02d}] "
            f"{group:9s} {name:45s} "
            f"N={weights.numel():,}"
        )

        fit_values = sample_values(
            weights,
            FIT_SAMPLE_SIZE,
            generator,
        )

        ternary_params = fit_ternary(fit_values)
        level4_params = fit_4level_symmetric(fit_values)
        uniform_params = fit_uniform_int2(fit_values)

        schemes = [
            (
                "ternary",
                quantize_ternary(weights, ternary_params),
                ternary_params,
            ),
            (
                "4level_opt",
                quantize_4level_symmetric(
                    weights,
                    level4_params,
                ),
                level4_params,
            ),
            (
                "uniform_int2",
                quantize_uniform_int2(
                    weights,
                    uniform_params,
                ),
                uniform_params,
            ),
        ]

        for scheme, q, params in schemes:
            result = metrics(weights, q)

            row = {
                "group": group,
                "tensor": name,
                "scheme": scheme,
                "n_weights": weights.numel(),
                **result,
            }

            for key, value in params.items():
                if key != "mse":
                    row[key] = value

            if scheme == "ternary":
                row["zero_fraction"] = (
                    (q == 0).float().mean().item()
                )

            rows.append(row)

    df = pd.DataFrame(rows)

    print("\n================ PER-TENSOR RESULTS ================\n")

    columns = [
        "group",
        "tensor",
        "scheme",
        "n_weights",
        "mse",
        "rmse",
        "mae",
        "max_error",
        "cosine",
        "sqnr_db",
    ]

    extra = [
        c
        for c in [
            "zero_fraction",
            "delta",
            "alpha",
            "beta",
            "threshold",
            "scale",
        ]
        if c in df.columns
    ]

    print(
        df[columns + extra]
        .round(7)
        .to_string(index=False)
    )

    # Weighted aggregate by group/scheme.
    # MSE/MAE are correctly weighted by number of weights.
    aggregate_rows = []

    for (group, scheme), part in df.groupby(
        ["group", "scheme"]
    ):
        n = part["n_weights"].to_numpy(dtype=np.float64)
        total = n.sum()

        weighted_mse = np.sum(part["mse"] * n) / total
        weighted_mae = np.sum(part["mae"] * n) / total

        # Cosine and max-error are not linearly composable.
        # Recompute exact global metrics below.
        original_parts = []
        quantized_parts = []

        for tensor_name in part["tensor"]:
            original = next(
                w
                for g, n_, w in tensors
                if g == group and n_ == tensor_name
            )

            row = part[part["tensor"] == tensor_name].iloc[0]

            fit_values = sample_values(
                original,
                FIT_SAMPLE_SIZE,
                generator,
            )

            if scheme == "ternary":
                params = {
                    "delta": row["delta"],
                    "alpha": row["alpha"],
                }
                q = quantize_ternary(original, params)

            elif scheme == "4level_opt":
                params = {
                    "alpha": row["alpha"],
                    "beta": row["beta"],
                    "threshold": row["threshold"],
                }
                q = quantize_4level_symmetric(
                    original,
                    params,
                )

            else:
                params = {
                    "scale": row["scale"],
                }
                q = quantize_uniform_int2(
                    original,
                    params,
                )

            original_parts.append(original)
            quantized_parts.append(q)

        original_all = torch.cat(original_parts)
        quantized_all = torch.cat(quantized_parts)

        exact = metrics(original_all, quantized_all)

        aggregate_rows.append(
            {
                "group": group,
                "scheme": scheme,
                "n_weights": int(total),
                "mse": weighted_mse,
                "rmse": math.sqrt(weighted_mse),
                "mae": weighted_mae,
                "max_error": exact["max_error"],
                "cosine": exact["cosine"],
                "sqnr_db": exact["sqnr_db"],
            }
        )

    summary = pd.DataFrame(aggregate_rows)

    print("\n================ AGGREGATED RESULTS ================\n")

    print(
        summary
        .sort_values(["group", "mse"])
        .round(7)
        .to_string(index=False)
    )

    print("\n================ BEST SCHEME PER GROUP ================\n")

    best = (
        summary
        .sort_values("mse")
        .groupby("group", as_index=False)
        .first()
    )

    print(
        best[
            [
                "group",
                "scheme",
                "n_weights",
                "mse",
                "cosine",
                "sqnr_db",
            ]
        ]
        .round(7)
        .to_string(index=False)
    )

    df.to_csv(
        "gpt2_quantization_per_tensor.csv",
        index=False,
    )

    summary.to_csv(
        "gpt2_quantization_summary.csv",
        index=False,
    )

    print("\nSaved:")
    print("  gpt2_quantization_per_tensor.csv")
    print("  gpt2_quantization_summary.csv")


if __name__ == "__main__":
    main()
