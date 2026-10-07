from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import prune
from tensors import load_model, total_parameters

import pdb


def generate_all_prune_outputs(
    model_path: str | Path,
    test_ratio: float = 0.50,
    p_norm: float = 2.0,
    sweep_ratios: tuple[float, ...] = (0.0, 0.25, 0.50, 0.75),
) -> dict[str, Any]:
    """Execute every function in prune.py and collect their outputs."""
    model = load_model(model_path)
    tensors = model["tensors"]


    helpers_output: dict[str, Any] = {
        "drop_counts_at_test_ratio": {},
        "channel_group_scores": {},
    }

    for t in tensors:
        # _drop_count test
        helpers_output["drop_counts_at_test_ratio"][t.name] = {
            "parameters_total": t.parameters,
            "drop_count_elements": prune._drop_count(t.parameters, test_ratio),
            "channels_total": t.channels,
            "drop_count_channels": prune._drop_count(t.channels, test_ratio),
        }
        # _group_scores test (L^p channel norms)
        helpers_output["channel_group_scores"][t.name] = {
            "p": p_norm,
            "scores": list(prune._group_scores(t, p=p_norm)),
        }


    fine_grained_output: dict[str, Any] = {}
    fine_pruned_tensors = []

    for t in tensors:
        mask = prune.magnitude_mask(t, test_ratio)
        masked_t = prune.apply_mask(t, mask)
        fine_pruned_tensors.append(masked_t)

        fine_grained_output[t.name] = {
            "ratio": test_ratio,
            "mask_kept_count": sum(mask),
            "mask_zeroed_count": len(mask) - sum(mask),
            # Store full mask as a list of 1s and 0s
            "mask": list(mask),
            "tensor_after": {
                "shape": list(masked_t.shape),
                "parameters": masked_t.parameters,
                "dtype": masked_t.dtype,
                # Sample first 10 values to keep the JSON readable
                "sample_data_head_10": list(masked_t.data[:10]),
            },
        }


    structured_output: dict[str, Any] = {}
    channel_pruned_tensors = []

    for t in tensors:
        keep = prune.channel_keep(t, test_ratio, p=p_norm)
        dropped_t = prune.drop_channels(t, keep)
        channel_pruned_tensors.append(dropped_t)

        structured_output[t.name] = {
            "ratio": test_ratio,
            "p": p_norm,
            "channels_before": t.channels,
            "channels_kept": list(keep),
            "channels_dropped_count": t.channels - len(keep),
            "tensor_after": {
                "shape": list(dropped_t.shape),
                "parameters": dropped_t.parameters,
                "dtype": dropped_t.dtype,
                "sample_data_head_10": list(dropped_t.data[:10]),
            },
        }

    bytes_stored_output: dict[str, Any] = {
        "original_dense": prune.bytes_stored(tensors, storage="dense"),
        "fine_pruned": {
            "masked_framework": prune.bytes_stored(
                fine_pruned_tensors, storage="masked", mask_encoding="framework"
            ),
            "masked_bitmap": prune.bytes_stored(
                fine_pruned_tensors, storage="masked", mask_encoding="bitmap"
            ),
            "sparse": prune.bytes_stored(fine_pruned_tensors, storage="sparse"),
            "dense": prune.bytes_stored(fine_pruned_tensors, storage="dense"),
        },
        "channel_pruned": {
            "dense": prune.bytes_stored(channel_pruned_tensors, storage="dense"),
            "sparse": prune.bytes_stored(channel_pruned_tensors, storage="sparse"),
        },
    }

    classify_removal_output = {
        "baseline_unpruned": prune.classify_removal(tensors, tensors, storage="dense"),
        "fine_pruned_masked": prune.classify_removal(
            tensors, fine_pruned_tensors, storage="masked"
        ),
        "fine_pruned_sparse": prune.classify_removal(
            tensors, fine_pruned_tensors, storage="sparse"
        ),
        "channel_pruned_dense": prune.classify_removal(
            tensors, channel_pruned_tensors, storage="dense"
        ),
    }


    sparsity_row_output = {
        "fine_row_sample": prune.sparsity_row(
            model=model["name"],
            ratio=test_ratio,
            granularity="fine",
            before=tensors,
            after=fine_pruned_tensors,
            storage="masked",
            mask_encoding="framework",
        ),
        "channel_row_sample": prune.sparsity_row(
            model=model["name"],
            ratio=test_ratio,
            granularity="channel",
            before=tensors,
            after=channel_pruned_tensors,
            storage="dense",
        ),
    }


    sweep_output = prune.sweep_model(
        model=model,
        ratios=sweep_ratios,
        granularities=prune.GRANULARITIES,
        p=p_norm,
    )

    return {
        "model_metadata": {
            "name": model["name"],
            "seed": model["seed"],
            "dtype": model["dtype"],
            "total_parameters_original": total_parameters(tensors),
            "layers": [t.name for t in tensors],
        },
        "functions": {
            "helpers": helpers_output,
            "magnitude_mask_and_apply_mask": fine_grained_output,
            "channel_keep_and_drop_channels": structured_output,
            "bytes_stored": bytes_stored_output,
            "classify_removal": classify_removal_output,
            "sparsity_row": sparsity_row_output,
            "sweep_model": sweep_output,
        },
    }

if __name__ == "__main__":
    model_path = Path("model.json")
    test_ratio = 0.50
    p_norm = 2.0
    sweep_ratios = (0.0, 0.25, 0.50, 0.75)

    outputs = generate_all_prune_outputs(
        model_path=model_path,
        test_ratio=test_ratio,
        p_norm=p_norm,
        sweep_ratios=sweep_ratios,
    )

    # Save outputs to a JSON file
    output_file = Path("prune_outputs.json")
    with open(output_file, "w") as f:
        json.dump(outputs, f, indent=4)

    print(f"Pruning outputs saved to {output_file.resolve()}")