#!/usr/bin/env python3
"""Fit one member of the 4 x 3 x 3 x 5 Xenium STORMI grid."""

from __future__ import annotations

import argparse
import functools
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import scanpy as sc
import stormi
from stormi.guides import AmortizedNormal
from stormi.model_input import prepare_model_input
from stormi.models.ATAC_RNA_pert import DEFAULT_MODULES as BASE_MODULES
from stormi.models.ATAC_RNA_pert import _model_impl as base_model
from stormi.models.ATAC_RNA_pert_cc import DEFAULT_MODULES as CC_MODULES
from stormi.models.ATAC_RNA_pert_cc import _model_impl as cc_model
from stormi.models.modules.priors import sample_cell_cycle_parameters_v1
from stormi.posterior import extract_latents_minibatched


HERE = Path(__file__).resolve().parent
STORMI_ROOT = Path("/g/stegle/tovar/apps/stormi_cc_env")
STORMI_COMMIT = "6a740f5c1b526d179a773365f8830a9f24934210"
DATA_PATH = Path(
    "/g/stegle/aivazidis/data/glioma_xenium/"
    "rna_metacells_leiden_23072026nn20res180.h5ad"
)
TF_PATH = HERE / "inputs/Human_TFs_all.txt"
CC_GENES_PATH = HERE / "inputs/regev_lab_cell_cycle_genes.txt"
INFLOW_PATH = HERE / "inputs/inflow_covariates.parquet"
DEFAULT_OUTPUT = Path("/scratch/tovar/xenium_trajectory_stability_180_runs")

CLUSTERS = ["4", "0", "21", "1", "6", "7"]
INITIAL_STATES = ["4"]
TERMINAL_STATES = ["1", "6", "7"]
CLUSTER_HOURS = {
    "4": 0.000000,
    "0": 9.721263,
    "21": 19.870785,
    "1": 31.434881,
    "6": 38.620992,
    "7": 40.000000,
}
ARMS = [
    ("no_inflow_no_prior", False, False),
    ("no_inflow_manual", False, True),
    ("inflow_no_prior", True, False),
    ("inflow_manual", True, True),
]
COUNT_MODES = ["cyto_nuclear_split", "nuclear_only", "cytoplasmic_only"]
CONFIGS = {
    "baseline": None,
    "cc_low": -5.0,
    "cc_high": -3.0,
}
SEEDS = [1, 2, 3, 4, 5]
ENV_COLUMNS = [f"stormi_env_inflow_factor_{i}" for i in range(1, 5)]
POSTERIOR_SITES = [
    "T_c",
    "detection_y_c",
    "z_pw",
    "z_eff",
    "loc_pw",
    "path_weights",
    "T_p",
]


def decode_task(task_id: int) -> dict[str, Any]:
    """Map Slurm task 0..179 to one experiment setting."""
    if not 0 <= task_id < 180:
        raise ValueError("task_id must be between 0 and 179")
    arm_index, within_arm = divmod(task_id, 45)
    mode_index, within_mode = divmod(within_arm, 15)
    config_index, seed_index = divmod(within_mode, 5)
    arm, use_inflow, use_time_prior = ARMS[arm_index]
    return {
        "task_id": task_id,
        "arm": arm,
        "use_inflow": use_inflow,
        "use_time_prior": use_time_prior,
        "count_mode": COUNT_MODES[mode_index],
        "config": list(CONFIGS)[config_index],
        "seed": SEEDS[seed_index],
    }


def check_stormi_checkout() -> None:
    imported_from = Path(stormi.__file__).resolve()
    if STORMI_ROOT not in imported_from.parents:
        raise RuntimeError(f"Imported STORMI from {imported_from}, not {STORMI_ROOT}")
    commit = subprocess.check_output(
        ["git", "-C", str(STORMI_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        [
            "git",
            "-C",
            str(STORMI_ROOT),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        text=True,
    ).strip()
    if commit != STORMI_COMMIT or dirty:
        raise RuntimeError(
            f"STORMI must be clean at {STORMI_COMMIT}; "
            f"found commit={commit}, modified={bool(dirty)}"
        )


def load_data(task: dict[str, Any]) -> Any:
    adata = sc.read_h5ad(DATA_PATH)
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    keep = adata.obs["leiden"].astype(str).isin(CLUSTERS)
    adata = adata[keep].copy()
    adata.obs["leiden"] = pd.Categorical(
        adata.obs["leiden"].astype(str), categories=CLUSTERS, ordered=True
    )
    if adata.n_obs != 30_264:
        raise ValueError(f"Expected 30,264 metacells, found {adata.n_obs:,}")

    if task["use_time_prior"]:
        adata.obs["hours_float_manual"] = (
            adata.obs["leiden"].astype(str).map(CLUSTER_HOURS).astype(np.float32)
        )

    if task["use_inflow"]:
        factors = pd.read_parquet(INFLOW_PATH).set_index("metacell_id")
        factors.index = factors.index.astype(str)
        factors = factors.reindex(adata.obs_names.astype(str))
        raw_columns = [f"Factor {i}" for i in range(1, 5)]
        if factors[raw_columns].isna().any().any():
            raise ValueError("Inflow factors do not cover all modeled metacells")
        for raw, target in zip(raw_columns, ENV_COLUMNS):
            adata.obs[target] = factors[raw].to_numpy(dtype=np.float32)

    nuclear = adata.layers["nuclear"].copy()
    cytoplasmic = adata.layers["cyto"].copy()
    mode = task["count_mode"]
    if mode == "cyto_nuclear_split":
        adata.X = nuclear
        adata.layers.pop("counts", None)
        adata.layers["unspliced"] = nuclear.copy()
        adata.layers["spliced"] = cytoplasmic.copy()
        adata.obs["has_rna"] = False
        adata.obs["has_splice"] = True
        adata.obs.pop("RNA counts", None)
    else:
        counts = nuclear if mode == "nuclear_only" else cytoplasmic
        adata.X = counts
        adata.layers["counts"] = counts.copy()
        adata.layers.pop("unspliced", None)
        adata.layers.pop("spliced", None)
        adata.obs["has_rna"] = True
        adata.obs["has_splice"] = False
        adata.obs["RNA counts"] = np.asarray(counts.sum(axis=1)).ravel()
    return adata


def make_model_and_input(
    adata: Any, task: dict[str, Any]
) -> tuple[Any, dict[str, Any]]:
    tf_list = pd.read_csv(TF_PATH, header=None).iloc[:, 0].astype(str).tolist()
    input_options: dict[str, Any] = {}
    if task["use_inflow"]:
        input_options.update(
            batch_annotation_level1="sample_id",
            env_covariate_cols=ENV_COLUMNS,
        )
    model_input = prepare_model_input(
        adata,
        tf_list,
        n_cells_col="n_cells",
        prior_time_col=("hours_float_manual" if task["use_time_prior"] else None),
        prior_timespan=100.0,
        terminal_states=TERMINAL_STATES,
        initial_states=INITIAL_STATES,
        cluster_key="leiden",
        **input_options,
    )
    # prepare_model_input computes this before it fills has_splice.
    model_input["splice"] = bool(np.any(model_input["has_splice"]))

    w_loc = CONFIGS[task["config"]]
    if w_loc is None:
        return functools.partial(base_model, modules=dict(BASE_MODULES)), model_input

    genes = adata.var_names.astype(str)
    cc_genes = CC_GENES_PATH.read_text().splitlines()
    s_mask = np.asarray(genes.isin(cc_genes[:43]), dtype=bool)
    g2m_mask = np.asarray(genes.isin(cc_genes[43:]), dtype=bool)
    if not s_mask.any() or not g2m_mask.any():
        raise ValueError("No S-phase or G2/M genes were found in the input")

    def configured_cc_prior(**kwargs: Any) -> dict[str, Any]:
        return sample_cell_cycle_parameters_v1(
            **kwargs,
            h_loc_low=0.0,
            h_loc_high=1.0,
            h_scale_low=0.1,
            h_scale_high=0.5,
            w_loc=w_loc,
            w_scale=0.5,
        )

    modules = dict(CC_MODULES)
    modules["sample_cell_cycle_parameters"] = configured_cc_prior
    model_input.update(
        use_cell_cycle=True,
        s_mask=s_mask,
        g2m_mask=g2m_mask,
        n_s=int(s_mask.sum()),
        n_g2m=int(g2m_mask.sum()),
    )
    return functools.partial(cc_model, modules=modules), model_input


def fix_split_guide_hvg_indexing() -> None:
    """Select both flattened splice channels for every chosen HVG."""
    import stormi.guides.api as guide_api
    import stormi.guides.nn as guide_nn

    original_selector = guide_api._compute_hvg_idx_from_tensor
    original_rna_features = guide_nn.rna_features

    def select_both_channels(data: Any, metacell_size: Any, n_top: int) -> np.ndarray:
        genes = np.asarray(
            original_selector(data, metacell_size, n_top), dtype=np.int32
        )
        return np.column_stack((2 * genes, 2 * genes + 1)).ravel()

    def matching_rna_features(
        data: Any, metacell_size: Any, *, hvg_idx: Any = None
    ) -> tuple[Any, Any, Any]:
        if hvg_idx is None:
            return original_rna_features(data, metacell_size, hvg_idx=None)
        genes = jnp.asarray(hvg_idx)[::2] // 2
        features, library, raw_library = original_rna_features(
            data, metacell_size, hvg_idx=genes
        )
        return jnp.repeat(features, 2, axis=1), library, raw_library

    guide_api._compute_hvg_idx_from_tensor = select_both_channels
    guide_nn.rna_features = matching_rna_features


def save_outputs(
    run_dir: Path,
    adata: Any,
    task: dict[str, Any],
    model: Any,
    model_input: dict[str, Any],
    training: dict[str, Any],
    posterior: dict[str, np.ndarray],
) -> None:
    time = np.asarray(posterior["T_c"], dtype=float).ravel()
    time_z = (time - time.mean()) / time.std()
    weights = np.asarray(posterior["path_weights"], dtype=float)
    hard_path = weights.argmax(axis=1)
    entropy = -(weights * np.log(weights + 1e-12)).sum(axis=1)

    cells = pd.DataFrame(
        {
            "cell_id": adata.obs_names.astype(str),
            "leiden": adata.obs["leiden"].astype(str).to_numpy(),
            "T_c": time,
            "posterior_time_z": time_z,
            "hard_terminal_state": np.asarray(TERMINAL_STATES)[hard_path],
            "max_path_weight": weights.max(axis=1),
            "path_entropy": entropy,
        }
    )
    for index, terminal in enumerate(TERMINAL_STATES):
        cells[f"path_terminal_{terminal}_weight"] = weights[:, index]
    cells.to_csv(run_dir / "cells.csv.gz", index=False, compression="gzip")

    losses = np.asarray(training["losses"], dtype=float)
    pd.Series(losses, name="loss").to_csv(
        run_dir / "training_loss.csv", index_label="step"
    )
    np.savez_compressed(
        run_dir / "latents_compact.npz",
        cell_id=np.asarray(adata.obs_names.astype(str), dtype=str),
        **{name: np.asarray(values) for name, values in posterior.items()},
    )
    stormi.save(
        run_dir / "stormi_model.pkl",
        model,
        model_input,
        training,
        adata,
        cell_type_key="leiden",
        save_grn=False,
        posterior=posterior,
        seed=task["seed"],
        cell_batch_size=500,
    )
    metadata = {
        **task,
        "stormi_commit": STORMI_COMMIT,
        "data_path": str(DATA_PATH),
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "terminal_states": TERMINAL_STATES,
        "training": {
            "warmup_steps": 10_000,
            "max_iterations": 50_000,
            "cell_batch_size": 500,
            "min_lr": 0.0005,
            "max_lr": 0.002,
            "patience": 3_000,
        },
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (run_dir / "SUCCESS").write_text("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_id", type=int, help="array index from 0 to 179")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    task = decode_task(args.task_id)
    seed = task["seed"]
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    check_stormi_checkout()
    if not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("A JAX GPU is required")

    run_dir = (
        args.output_root
        / task["arm"]
        / "runs"
        / task["count_mode"]
        / task["config"]
        / f"seed{seed:02d}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    print(json.dumps({**task, "run_dir": str(run_dir)}, indent=2), flush=True)

    adata = load_data(task)
    model, model_input = make_model_and_input(adata, task)
    if task["count_mode"] == "cyto_nuclear_split":
        fix_split_guide_hvg_indexing()

    guide = AmortizedNormal(model, model_input, init_seed=seed, hvg_n_top=3_000)
    guide.warm_up(model_input, n_steps=10_000, seed=seed, max_cells=500)
    training = stormi.train(
        model,
        guide,
        model_input=model_input,
        seed=seed,
        max_iterations=50_000,
        min_lr=0.0005,
        max_lr=0.002,
        ramp_up_fraction=0.1,
        log_interval=1_000,
        patience=3_000,
        cell_batch_size=500,
        grad_clip_norm=1.0,
        stratified_training=False,
    )

    sites = list(POSTERIOR_SITES)
    if task["config"] != "baseline":
        sites += ["w_cf", "h_fg_s", "h_fg_g2m"]
    extracted = extract_latents_minibatched(
        model,
        model_input,
        training,
        return_sites=sites,
        seed=seed,
        cell_batch_size=500,
    )
    posterior = {name: np.asarray(extracted[name]) for name in sites}
    if any(not np.isfinite(value).all() for value in posterior.values()):
        raise ValueError("Posterior contains non-finite values")
    save_outputs(run_dir, adata, task, model, model_input, training, posterior)


if __name__ == "__main__":
    main()
