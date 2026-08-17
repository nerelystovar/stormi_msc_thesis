"""Compare the completed K=4, K=5 and K=6 inflow MOFA-FLEX runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import linear_sum_assignment


def _factor_columns(table: pd.DataFrame) -> list[str]:
    """Return ``Factor`` columns in numeric order."""

    factors = [str(column) for column in table if str(column).startswith("Factor ")]
    return sorted(factors, key=lambda factor: int(factor.rsplit(" ", 1)[1]))


def _weight_matrix(weights: pd.DataFrame) -> pd.DataFrame:
    required = {"view", "feature", "factor", "weight"}
    missing = required.difference(weights.columns)
    if missing:
        raise ValueError(f"all_weights is missing columns: {sorted(missing)}")
    matrix = weights.pivot(
        index=["view", "feature"], columns="factor", values="weight"
    ).fillna(0.0)
    return matrix[_factor_columns(matrix)].sort_index()


def _factor_cosines(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
) -> pd.DataFrame:
    features = reference.index.union(candidate.index)
    a = reference.reindex(features, fill_value=0.0).to_numpy(dtype=float)
    b = candidate.reindex(features, fill_value=0.0).to_numpy(dtype=float)
    denominator = np.outer(np.linalg.norm(a, axis=0), np.linalg.norm(b, axis=0))
    values = np.divide(
        a.T @ b,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    return pd.DataFrame(values, index=reference.columns, columns=candidate.columns)


def _top_features(
    weights: pd.DataFrame,
    factor: str,
    n: int,
) -> set[tuple[str, str]]:
    table = weights.loc[
        (weights["factor"] == factor) & (weights["weight"] > 0),
        ["view", "feature", "weight"],
    ].nlargest(n, "weight")
    return set(zip(table["view"].astype(str), table["feature"].astype(str)))


def _score_correlations(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    reference_factor: str,
    candidate_factor: str,
    min_cells_per_sample: int,
) -> tuple[float, float, int]:
    scores = (
        reference[["metacell_id", "sample_id", reference_factor]]
        .rename(columns={reference_factor: "reference"})
        .merge(
            candidate[["metacell_id", candidate_factor]].rename(
                columns={candidate_factor: "candidate"}
            ),
            on="metacell_id",
            validate="one_to_one",
        )
    )
    if len(scores) != len(reference) or len(scores) != len(candidate):
        raise ValueError("Factor-score tables contain different metacells")

    global_correlation = np.nan
    if scores["reference"].nunique() > 1 and scores["candidate"].nunique() > 1:
        global_correlation = scores["reference"].corr(
            scores["candidate"], method="spearman"
        )
    sample_correlations = []
    for _, sample in scores.groupby("sample_id", observed=True):
        if len(sample) < min_cells_per_sample:
            continue
        if sample["reference"].nunique() < 2 or sample["candidate"].nunique() < 2:
            continue
        correlation = sample["reference"].corr(sample["candidate"], method="spearman")
        if np.isfinite(correlation):
            sample_correlations.append(correlation)

    median_sample_correlation = (
        float(np.median(sample_correlations)) if sample_correlations else np.nan
    )
    return float(global_correlation), median_sample_correlation, len(sample_correlations)


def load_kgrid_runs(
    configs: dict[int, dict[str, Any]],
    input_path: str | Path,
) -> tuple[pd.DataFrame, dict[int, dict[int, dict[str, Any]]]]:
    """Load completed runs whose saved settings match the requested grid."""

    input_path = Path(input_path)
    status = []
    runs_by_k: dict[int, dict[int, dict[str, Any]]] = {}

    for k, config in sorted(configs.items()):
        seeds = [int(seed) for seed in config["seeds"]]
        runs_by_k[k] = {}

        for seed in seeds:
            run_name = f"{config['run_basename']}_seed{seed}"
            run_dir = Path(config["outdir"]) / run_name
            summary_path = run_dir / "summary.json"
            score_path = run_dir / "factor_scores.parquet"
            weight_path = run_dir / "all_weights.parquet"
            summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
            options = summary.get("mofaflex_options", {})
            filters = summary.get("filters", {})

            checks = {
                "input": Path(summary.get("lrdata_h5ad", "")) == input_path,
                "K": options.get("n_factors") == k,
                "seed": options.get("seed") == seed,
                "random_init": options.get("init_factors") == "random",
                "nonnegative_weights": options.get("nonnegative_weights") is True,
                "nonnegative_factors": options.get("nonnegative_factors") is True,
                "no_guidance": options.get("guiding_vars") == [],
                "prevalence": filters.get("min_prevalence") == 0.01,
                "total_cap": filters.get("max_total_features") == 0,
                "sender_cap": filters.get("max_features_per_sender") == 300,
                "minimum_sender_features": filters.get("min_features_per_sender") == 5,
            }
            files_exist = score_path.exists() and weight_path.exists()
            complete = (
                summary.get("status") == "complete"
                and files_exist
                and all(checks.values())
            )
            status.append(
                {
                    "K": k,
                    "seed": seed,
                    "run_name": run_name,
                    "status": summary.get("status", "missing"),
                    "config_ok": all(checks.values()),
                    "files_exist": files_exist,
                    "failed_checks": ", ".join(
                        name for name, passed in checks.items() if not passed
                    ),
                }
            )
            if not complete:
                continue

            scores = pd.read_parquet(score_path)
            weights = pd.read_parquet(weight_path)
            factors = _factor_columns(scores)
            if len(factors) != k:
                raise ValueError(f"{run_name}: expected {k} factors, found {factors}")
            if scores["metacell_id"].duplicated().any():
                raise ValueError(f"{run_name}: metacell_id is not unique")
            if scores[factors].min().min() < -1e-8 or weights["weight"].min() < -1e-8:
                raise ValueError(f"{run_name}: negative NMF values found")

            runs_by_k[k][seed] = {
                "run_name": run_name,
                "summary": summary,
                "factor_scores": scores,
                "all_weights": weights,
            }

    return pd.DataFrame(status), runs_by_k


def analyse_kgrid_runs(
    runs_by_k: dict[int, dict[int, dict[str, Any]]],
    configs: dict[int, dict[str, Any]],
    min_cells_per_sample: int = 10,
    top_n_interactions: int = 100,
) -> tuple[dict[int, dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    """Select the loading medoid at each K and match its factors across seeds."""

    results: dict[int, dict[str, Any]] = {}
    all_matches = []
    summaries = []

    for k, config in sorted(configs.items()):
        seeds = [int(seed) for seed in config["seeds"]]
        runs = runs_by_k.get(k, {})
        missing = [seed for seed in seeds if seed not in runs]
        if missing:
            print(f"K={k}: missing complete validated seeds {missing}")
            continue

        matrices = {seed: _weight_matrix(runs[seed]["all_weights"]) for seed in seeds}
        pairwise = {seed: [] for seed in seeds}
        for position, seed_a in enumerate(seeds):
            for seed_b in seeds[position + 1 :]:
                cosines = _factor_cosines(matrices[seed_a], matrices[seed_b])
                rows, columns = linear_sum_assignment(-cosines.to_numpy())
                similarity = float(cosines.to_numpy()[rows, columns].mean())
                pairwise[seed_a].append(similarity)
                pairwise[seed_b].append(similarity)

        centrality = {
            seed: float(np.mean(values)) if values else 1.0
            for seed, values in pairwise.items()
        }
        reference_seed = max(centrality, key=centrality.get)
        reference_run = runs[reference_seed]
        reference_matrix = matrices[reference_seed]
        reference_factors = _factor_columns(reference_matrix)
        seed_labels = {seed: position + 1 for position, seed in enumerate(seeds)}
        top_features = {
            seed: {
                factor: _top_features(
                    runs[seed]["all_weights"], factor, top_n_interactions
                )
                for factor in _factor_columns(matrices[seed])
            }
            for seed in seeds
        }
        matches = []

        for seed in seeds:
            cosines = _factor_cosines(reference_matrix, matrices[seed])
            common_features = reference_matrix.index.union(matrices[seed].index)
            aligned_reference = reference_matrix.reindex(common_features, fill_value=0.0)
            aligned_candidate = matrices[seed].reindex(common_features, fill_value=0.0)
            rows, columns = linear_sum_assignment(-cosines.to_numpy())
            alignment = {
                str(cosines.index[row]): str(cosines.columns[column])
                for row, column in zip(rows, columns)
            }
            for reference_factor in reference_factors:
                candidate_factor = alignment[reference_factor]
                global_score, sample_score, n_samples = _score_correlations(
                    reference_run["factor_scores"],
                    runs[seed]["factor_scores"],
                    reference_factor,
                    candidate_factor,
                    min_cells_per_sample,
                )
                reference_top = top_features[reference_seed][reference_factor]
                candidate_top = top_features[seed][candidate_factor]
                union = reference_top | candidate_top
                jaccard = len(reference_top & candidate_top) / len(union) if union else np.nan
                row = {
                    "K": k,
                    "reference_seed": reference_seed,
                    "reference_seed_label": seed_labels[reference_seed],
                    "seed": seed,
                    "seed_label": seed_labels[seed],
                    "is_self_comparison": seed == reference_seed,
                    "reference_factor": reference_factor,
                    "matched_factor": candidate_factor,
                    "weight_cosine": float(cosines.loc[reference_factor, candidate_factor]),
                    "weight_spearman": float(
                        aligned_reference[reference_factor].corr(
                            aligned_candidate[candidate_factor], method="spearman"
                        )
                    )
                    if aligned_reference[reference_factor].nunique() > 1
                    and aligned_candidate[candidate_factor].nunique() > 1
                    else np.nan,
                    "global_score_spearman": global_score,
                    "median_sample_score_spearman": sample_score,
                    "n_valid_samples": n_samples,
                    "top_interaction_jaccard": jaccard,
                    "passes_all": (
                        cosines.loc[reference_factor, candidate_factor] >= 0.80
                        and sample_score >= 0.80
                        and jaccard >= 0.50
                    ),
                }
                matches.append(row)
                all_matches.append(row)

        matches = pd.DataFrame(matches)
        independent = matches.loc[~matches["is_self_comparison"]]
        for factor in reference_factors:
            factor_matches = independent.loc[independent["reference_factor"] == factor]
            n_runs = factor_matches["seed"].nunique()
            required_passes = max(1, int(np.ceil(0.75 * n_runs)))
            n_passing = int(factor_matches["passes_all"].sum())
            summaries.append(
                {
                    "K": k,
                    "reference_seed": reference_seed,
                    "reference_seed_label": seed_labels[reference_seed],
                    "reference_factor": factor,
                    "independent_runs": n_runs,
                    "independent_runs_passing_all": n_passing,
                    "stable": n_passing >= required_passes,
                    "median_weight_cosine": factor_matches["weight_cosine"].median(),
                    "minimum_weight_cosine": factor_matches["weight_cosine"].min(),
                    "median_weight_spearman": factor_matches[
                        "weight_spearman"
                    ].median(),
                    "median_global_score_spearman": factor_matches[
                        "global_score_spearman"
                    ].median(),
                    "median_within_sample_spearman": factor_matches[
                        "median_sample_score_spearman"
                    ].median(),
                    "median_top_interaction_jaccard": factor_matches[
                        "top_interaction_jaccard"
                    ].median(),
                }
            )

        results[k] = {
            "reference_seed": reference_seed,
            "reference_seed_label": seed_labels[reference_seed],
            "seed_labels": seed_labels,
            "reference_run": reference_run,
            "reference_factors": reference_factors,
            "mean_seed_similarity": centrality,
            "matches": matches,
            "top_n_interactions": top_n_interactions,
        }

    return results, pd.DataFrame(all_matches), pd.DataFrame(summaries)


def plot_seed_reproducibility_heatmaps(
    results: dict[int, dict[str, Any]],
) -> None:
    """Plot fixed-K reproducibility; raw seeds are displayed as 1, 2, ... ."""

    for k, result in sorted(results.items()):
        matches = result["matches"].loc[lambda table: ~table["is_self_comparison"]]
        if matches.empty:
            continue

        panels = [
            ("weight_cosine", "Loading cosine", 0, 1),
            ("weight_spearman", "Loading Spearman", -1, 1),
            (
                "top_interaction_jaccard",
                f"Top-{result['top_n_interactions']} interaction Jaccard",
                0,
                1,
            ),
        ]
        seeds = [seed for seed in result["seed_labels"] if seed != result["reference_seed"]]
        fig, axes = plt.subplots(1, 3, figsize=(16, max(4.5, 0.8 * k)))
        for ax, (metric, title, vmin, vmax) in zip(axes, panels):
            matrix = matches.pivot(
                index="reference_factor", columns="seed", values=metric
            ).reindex(index=result["reference_factors"], columns=seeds)
            matrix.columns = [result["seed_labels"][seed] for seed in matrix.columns]
            sns.heatmap(
                matrix,
                ax=ax,
                cmap="mako",
                vmin=vmin,
                vmax=vmax,
                annot=True,
                fmt=".2f",
                cbar_kws={"shrink": 0.75},
            )
            ax.set_title(title)
            ax.set_xlabel("Seed")
            ax.set_ylabel("Medoid factor" if ax is axes[0] else "")
            ax.tick_params(axis="x", labelrotation=0)
            ax.tick_params(axis="y", labelrotation=0)

        fig.suptitle(
            f"K={k}, medoid seed {result['reference_seed_label']} (self excluded)"
        )
        fig.tight_layout()
        plt.show()


def build_medoid_factor_matrices(
    results: dict[int, dict[str, Any]],
    min_cells_per_sample: int = 10,
    top_n_interactions: int = 100,
) -> dict[str, Any]:
    """Compare loading composition and factor scores across the K medoids."""

    if not results:
        raise ValueError("No complete K-specific seed panels are available")
    feature_hashes = {
        result["reference_run"]["summary"].get("selected_features_sha256")
        for result in results.values()
    }
    if None in feature_hashes or len(feature_hashes) != 1:
        raise ValueError("Cross-K comparisons require the same selected features")

    weights = []
    scores: pd.DataFrame | None = None
    labels = []
    top_sets = {}

    for k, result in sorted(results.items()):
        factors = result["reference_factors"]
        rename = {factor: f"K{k} | F{factor.rsplit(' ', 1)[1]}" for factor in factors}
        labels.extend(rename.values())
        run = result["reference_run"]
        weights.append(_weight_matrix(run["all_weights"])[factors].rename(columns=rename))
        for factor in factors:
            top_sets[rename[factor]] = _top_features(
                run["all_weights"], factor, top_n_interactions
            )

        score_part = run["factor_scores"][["metacell_id", "sample_id"] + factors].rename(
            columns=rename
        )
        if scores is None:
            scores = score_part
        else:
            scores = scores.merge(
                score_part.drop(columns="sample_id"),
                on="metacell_id",
                validate="one_to_one",
            )

    combined_weights = pd.concat(weights, axis=1).fillna(0.0)[labels]
    values = combined_weights.to_numpy(dtype=float)
    denominator = np.outer(np.linalg.norm(values, axis=0), np.linalg.norm(values, axis=0))
    weight_cosine = np.divide(
        values.T @ values,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    jaccard = [
        [
            len(top_sets[a] & top_sets[b]) / len(top_sets[a] | top_sets[b])
            if top_sets[a] | top_sets[b]
            else np.nan
            for b in labels
        ]
        for a in labels
    ]

    if scores is None:
        raise ValueError("No factor scores are available")
    sample_correlations = []
    for _, sample in scores.groupby("sample_id", observed=True):
        if len(sample) < min_cells_per_sample:
            continue
        variable = sample[labels].nunique() > 1
        matrix = pd.DataFrame(np.nan, index=labels, columns=labels)
        variable_labels = variable.index[variable].tolist()
        matrix.loc[variable_labels, variable_labels] = sample[variable_labels].corr(
            method="spearman"
        )
        sample_correlations.append(matrix.to_numpy())
    if not sample_correlations:
        raise ValueError("No sample has enough metacells for within-sample correlations")
    stack = np.stack(sample_correlations)
    valid_counts = np.isfinite(stack).sum(axis=0)
    median_sample = np.full((len(labels), len(labels)), np.nan)
    for row in range(len(labels)):
        for column in range(len(labels)):
            finite_values = stack[:, row, column]
            finite_values = finite_values[np.isfinite(finite_values)]
            if len(finite_values):
                median_sample[row, column] = np.median(finite_values)

    variable = scores[labels].nunique() > 1
    variable_labels = variable.index[variable].tolist()
    global_spearman = pd.DataFrame(np.nan, index=labels, columns=labels)
    global_spearman.loc[variable_labels, variable_labels] = scores[
        variable_labels
    ].corr(method="spearman")

    return {
        "weight_cosine": pd.DataFrame(weight_cosine, index=labels, columns=labels),
        "top_interaction_jaccard": pd.DataFrame(
            jaccard, index=labels, columns=labels
        ),
        "top_n_interactions": top_n_interactions,
        "global_score_spearman": global_spearman,
        "median_within_sample_score_spearman": pd.DataFrame(
            median_sample, index=labels, columns=labels
        ),
        "n_valid_samples": pd.DataFrame(
            valid_counts, index=labels, columns=labels
        ),
    }


def plot_medoid_factor_heatmaps(matrices: dict[str, Any]) -> None:
    """Plot the four cross-K medoid comparison matrices in one figure."""

    top_n = matrices.get("top_n_interactions", 100)
    panels = [
        ("weight_cosine", "Loading cosine", 0, 1),
        ("top_interaction_jaccard", f"Top-{top_n} interaction Jaccard", 0, 1),
        ("global_score_spearman", "Global score Spearman", -1, 1),
        (
            "median_within_sample_score_spearman",
            "Median within-sample score Spearman",
            -1,
            1,
        ),
    ]
    size = max(14, 0.9 * len(matrices["weight_cosine"]))
    fig, axes = plt.subplots(2, 2, figsize=(size, size), constrained_layout=True)

    for ax, (key, title, vmin, vmax) in zip(axes.ravel(), panels):
        matrix = matrices[key]
        sns.heatmap(
            matrix,
            ax=ax,
            cmap="mako",
            vmin=vmin,
            vmax=vmax,
            center=0 if vmin < 0 else None,
            square=True,
            annot=True,
            fmt=".2f",
            annot_kws={"fontsize": 6},
            cbar_kws={"shrink": 0.7},
        )
        ranks = [label.split(" | ", 1)[0] for label in matrix.index]
        for boundary in range(1, len(ranks)):
            if ranks[boundary] != ranks[boundary - 1]:
                ax.axhline(boundary, color="black", linewidth=1.2)
                ax.axvline(boundary, color="black", linewidth=1.2)
        ax.set_title(title)
        ax.tick_params(axis="x", labelrotation=90, labelsize=8)
        ax.tick_params(axis="y", labelrotation=0, labelsize=8)

    plt.show()
