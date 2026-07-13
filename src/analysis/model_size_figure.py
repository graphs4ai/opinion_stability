from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats


EXPERIMENTS = [
    {
        "id": "E1",
        "name": "Free summary",
        "path": "dados/experimento_1/respostas_iterativas.csv",
        "intervention": "summary",
    },
    {
        "id": "E2",
        "name": "E2 — Negative points and positive points",
        "path": "dados/experimento_2/respostas_iterativas_v2.csv",
        "intervention": "summary v2",
    },
    {
        "id": "E3",
        "name": "Trade-offs and uncertainty",
        "path": "dados/experimento_3/respostas_iterativas_v3.csv",
        "intervention": "structured summary",
    },
]

SHORT_MODEL_NAMES = {
    "Qwen/Qwen3-14B": "Qwen3 14B",
    "Qwen/Qwen3-30B-A3B": "Qwen3 30B",
    "deepseek-ai/DeepSeek-V4-Flash": "DeepSeek V4",
    "deepseek-ai/DeepSeek-V3.2": "DeepSeek V3.2",
    "google/gemma-3-4b-it": "Gemma 3 4B",
    "google/gemma-4-31B-it": "Gemma 4 31B",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": "Llama 4 Scout",
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "Llama 3.1 8B",
    "mistralai/Mistral-Nemo-Instruct-2407": "Mistral Nemo",
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506": "Mistral Small",
}

# Keep active and total parameter counts separate. Dense models use all
# parameters per token, while MoE models route only a subset of parameters; using
# a single size column would mix two different computational quantities.
MODEL_METADATA = {
    "Qwen/Qwen3-14B": {
        "family": "Qwen",
        "architecture": "Dense",
        "total_params_b": 14.0,
        "active_params_b": 14.0,
    },
    "Qwen/Qwen3-30B-A3B": {
        "family": "Qwen",
        "architecture": "MoE",
        "total_params_b": 30.0,
        "active_params_b": 3.0,
    },
    "google/gemma-3-4b-it": {
        "family": "Gemma",
        "architecture": "Dense",
        "total_params_b": 4.0,
        "active_params_b": 4.0,
    },
    "google/gemma-4-31B-it": {
        "family": "Gemma",
        "architecture": "Dense",
        "total_params_b": 31.0,
        "active_params_b": 31.0,
    },
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": {
        "family": "Llama",
        "architecture": "MoE",
        "total_params_b": 109.0,
        "active_params_b": 17.0,
    },
    "meta-llama/Meta-Llama-3.1-8B-Instruct": {
        "family": "Llama",
        "architecture": "Dense",
        "total_params_b": 8.0,
        "active_params_b": 8.0,
    },
    "mistralai/Mistral-Nemo-Instruct-2407": {
        "family": "Mistral",
        "architecture": "Dense",
        "total_params_b": 12.0,
        "active_params_b": 12.0,
    },
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506": {
        "family": "Mistral",
        "architecture": "Dense",
        "total_params_b": 24.0,
        "active_params_b": 24.0,
    },
    "deepseek-ai/DeepSeek-V4-Flash": {
        "family": "DeepSeek",
        "architecture": "MoE",
        "total_params_b": 284.0,
        "active_params_b": 13.0,
    },
    "deepseek-ai/DeepSeek-V3.2": {
        "family": "DeepSeek",
        "architecture": "MoE",
        "total_params_b": 671.0,
        "active_params_b": 37.0,
    },
}

FAMILY_COLORS = {
    "DeepSeek": "#2563eb",
    "Gemma": "#d97706",
    "Llama": "#7c3aed",
    "Mistral": "#059669",
    "Qwen": "#dc2626",
}
ARCHITECTURE_MARKERS = {"Dense": "o", "MoE": "^"}


@dataclass(frozen=True)
class BootstrapConfig:
    n_boot: int = 10_000
    seed: int = 20260712
    alpha: float = 0.05


def _short_model_name(model: str) -> str:
    return SHORT_MODEL_NAMES.get(model, str(model).split("/")[-1][:22])


def validate_model_metadata(models: pd.Series | list[str]) -> pd.DataFrame:
    observed = pd.Series(models, dtype="object").dropna()
    if observed.duplicated().any():
        duplicates = sorted(observed[observed.duplicated()].unique())
        raise ValueError(f"Duplicate model identifiers in model list: {duplicates}")

    observed_set = set(observed.tolist())
    metadata_set = set(MODEL_METADATA)
    missing = sorted(observed_set - metadata_set)
    if missing:
        raise ValueError(f"Missing model-size metadata for: {missing}")

    unused = sorted(metadata_set - observed_set)
    if unused:
        warnings.warn(
            f"Model-size metadata contains unused models: {unused}",
            stacklevel=2,
        )

    metadata = pd.DataFrame(
        [{"modelo": model, **values} for model, values in MODEL_METADATA.items()]
    )
    metadata = metadata[metadata["modelo"].isin(observed_set)].copy()

    required = ["family", "architecture", "total_params_b", "active_params_b"]
    missing_values = metadata[required].isna().any(axis=1)
    if missing_values.any():
        bad = metadata.loc[missing_values, "modelo"].tolist()
        raise ValueError(f"Missing required metadata values for: {bad}")

    valid_arch = {"Dense", "MoE"}
    bad_arch = metadata[~metadata["architecture"].isin(valid_arch)]
    if not bad_arch.empty:
        raise ValueError(
            "Invalid architecture values: "
            f"{bad_arch[['modelo', 'architecture']].to_dict('records')}"
        )

    for col in ["total_params_b", "active_params_b"]:
        non_positive = metadata[metadata[col] <= 0]
        if not non_positive.empty:
            raise ValueError(
                f"All {col} values must be positive: "
                f"{non_positive[['modelo', col]].to_dict('records')}"
            )

    bad_order = metadata[metadata["active_params_b"] > metadata["total_params_b"]]
    if not bad_order.empty:
        raise ValueError(
            "active_params_b must be <= total_params_b: "
            f"{bad_order[['modelo', 'active_params_b', 'total_params_b']].to_dict('records')}"
        )

    dense_mismatch = metadata[
        (metadata["architecture"] == "Dense")
        & ~np.isclose(metadata["active_params_b"], metadata["total_params_b"])
    ]
    if not dense_mismatch.empty:
        raise ValueError(
            "Dense models must have active_params_b == total_params_b: "
            f"{dense_mismatch[['modelo', 'active_params_b', 'total_params_b']].to_dict('records')}"
        )

    moe_without_sparsity = metadata[
        (metadata["architecture"] == "MoE")
        & (metadata["active_params_b"] >= metadata["total_params_b"])
    ]
    if not moe_without_sparsity.empty:
        raise ValueError(
            "MoE models must have active_params_b < total_params_b: "
            f"{moe_without_sparsity[['modelo', 'active_params_b', 'total_params_b']].to_dict('records')}"
        )

    if metadata["modelo"].duplicated().any():
        duplicates = metadata.loc[metadata["modelo"].duplicated(), "modelo"].tolist()
        raise ValueError(f"Duplicate metadata rows: {duplicates}")

    return metadata.reset_index(drop=True)


def _spearman_with_bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    config: BootstrapConfig = BootstrapConfig(),
) -> tuple[float, float, float, float, int]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    if n < 3 or len(np.unique(x)) < 3 or len(np.unique(y)) < 2:
        return np.nan, np.nan, np.nan, np.nan, n

    rho, p_value = stats.spearmanr(x, y)
    rng = np.random.default_rng(config.seed)
    boot = []
    for _ in range(config.n_boot):
        idx = rng.integers(0, n, size=n)
        bx = x[idx]
        by = y[idx]
        if len(np.unique(bx)) < 2 or len(np.unique(by)) < 2:
            continue
        brho, _ = stats.spearmanr(bx, by)
        if np.isfinite(brho):
            boot.append(brho)
    if not boot:
        return float(rho), float(p_value), np.nan, np.nan, n
    ci_low, ci_high = np.percentile(
        boot,
        [100 * config.alpha / 2, 100 * (1 - config.alpha / 2)],
    )
    return float(rho), float(p_value), float(ci_low), float(ci_high), n


def _theil_sen_line_and_band(
    x: np.ndarray,
    y: np.ndarray,
    grid: np.ndarray,
    config: BootstrapConfig = BootstrapConfig(n_boot=2_000),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_log = np.log10(np.asarray(x, dtype=float))
    y = np.asarray(y, dtype=float)
    grid_log = np.log10(np.asarray(grid, dtype=float))
    slope, intercept, _, _ = stats.theilslopes(y, x_log, 0.95)
    y_hat = intercept + slope * grid_log

    rng = np.random.default_rng(config.seed + 17)
    preds = []
    n = len(x_log)
    for _ in range(config.n_boot):
        idx = rng.integers(0, n, size=n)
        bx = x_log[idx]
        by = y[idx]
        if len(np.unique(bx)) < 2:
            continue
        try:
            bslope, bintercept, _, _ = stats.theilslopes(by, bx, 0.95)
        except Exception:
            continue
        pred = bintercept + bslope * grid_log
        if np.all(np.isfinite(pred)):
            preds.append(pred)

    if preds:
        band = np.vstack(preds)
        low, high = np.percentile(band, [2.5, 97.5], axis=0)
    else:
        low = np.full_like(grid, np.nan, dtype=float)
        high = np.full_like(grid, np.nan, dtype=float)
    return y_hat, low, high


def _prepare_size_data(trajectories: pd.DataFrame) -> pd.DataFrame:
    required = {"experiment", "modelo", "first_to_final_abs_delta"}
    missing = required - set(trajectories.columns)
    if missing:
        raise ValueError(f"Trajectories dataframe is missing columns: {sorted(missing)}")

    if trajectories[["experiment", "modelo", "first_to_final_abs_delta"]].isna().any().any():
        raise ValueError("Missing values found in required trajectory columns.")

    metadata = validate_model_metadata(sorted(trajectories["modelo"].unique()))
    model_protocol = (
        trajectories.groupby(["experiment", "modelo"], as_index=False)
        .agg(mean_abs_first_to_final_displacement=("first_to_final_abs_delta", "mean"))
    )
    aggregated = (
        model_protocol.groupby("modelo", as_index=False)
        .agg(mean_abs_first_to_final_displacement=("mean_abs_first_to_final_displacement", "mean"))
    )
    aggregated["scope"] = "Aggregated E1/E2/E3"
    by_experiment = model_protocol.rename(columns={"experiment": "scope"})
    scope_labels = {
        "E1": "Free summary",
        "E2": "E2 — Negative points and positive points",
        "E3": "Trade-offs and uncertainty",
    }
    by_experiment["scope"] = by_experiment["scope"].map(scope_labels)
    data = pd.concat(
        [
            by_experiment[["scope", "modelo", "mean_abs_first_to_final_displacement"]],
            aggregated[["scope", "modelo", "mean_abs_first_to_final_displacement"]],
        ],
        ignore_index=True,
    )
    data = data.merge(metadata, on="modelo", how="left", validate="many_to_one")
    data["model_short"] = data["modelo"].map(_short_model_name)
    if data.isna().any().any():
        missing_cols = data.columns[data.isna().any()].tolist()
        raise ValueError(f"Missing values after metadata merge: {missing_cols}")
    return data


def compute_model_size_statistics(
    trajectories: pd.DataFrame,
    config: BootstrapConfig = BootstrapConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = _prepare_size_data(trajectories)
    rows = []
    for scope, sub in data.groupby("scope", sort=False):
        for metric, label in [
            ("active_params_b", "Active parameters per token"),
            ("total_params_b", "Total parameters"),
        ]:
            rho, p_value, ci_low, ci_high, n = _spearman_with_bootstrap_ci(
                sub[metric].to_numpy(),
                sub["mean_abs_first_to_final_displacement"].to_numpy(),
                config,
            )
            rows.append(
                {
                    "scope": scope,
                    "size_metric": metric,
                    "size_metric_label": label,
                    "spearman_rho": rho,
                    "p_value": p_value,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "n": n,
                }
            )
    stats_table = pd.DataFrame(rows)

    agg = data[data["scope"] == "Aggregated E1/E2/E3"].copy()
    loo_rows = []
    for model in sorted(agg["modelo"].unique()):
        sub = agg[agg["modelo"] != model]
        rho, p_value, ci_low, ci_high, n = _spearman_with_bootstrap_ci(
            sub["active_params_b"].to_numpy(),
            sub["mean_abs_first_to_final_displacement"].to_numpy(),
            config,
        )
        loo_rows.append(
            {
                "removed_model": model,
                "removed_model_short": _short_model_name(model),
                "spearman_rho": rho,
                "p_value": p_value,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "n": n,
            }
        )
    leave_one_out = pd.DataFrame(loo_rows).sort_values("spearman_rho")
    return stats_table, leave_one_out


def _format_p_value(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "NA"
    if p_value < 0.001:
        return f"{p_value:.1e}"
    return f"{p_value:.3f}"


def _label_extreme_models(data: pd.DataFrame) -> set[str]:
    labels = set(data.nlargest(2, "mean_abs_first_to_final_displacement")["modelo"])
    labels |= set(data.nsmallest(1, "mean_abs_first_to_final_displacement")["modelo"])
    labels |= {"Qwen/Qwen3-30B-A3B", "deepseek-ai/DeepSeek-V3.2"}
    return labels


def _plot_size_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    stats_row: pd.Series,
    metric: str,
    title: str,
    panel_label: str,
    y_limits: tuple[float, float],
) -> None:
    x = data[metric].to_numpy(dtype=float)
    y = data["mean_abs_first_to_final_displacement"].to_numpy(dtype=float)
    grid = np.logspace(np.log10(x.min() * 0.9), np.log10(x.max() * 1.1), 160)
    trend, low, high = _theil_sen_line_and_band(x, y, grid)

    ax.fill_between(grid, low, high, color="#94a3b8", alpha=0.22, linewidth=0)
    ax.plot(grid, trend, color="#334155", linewidth=2.0, label="Theil–Sen trend")

    for architecture, marker in ARCHITECTURE_MARKERS.items():
        arch_sub = data[data["architecture"] == architecture]
        for family, fam_sub in arch_sub.groupby("family"):
            ax.scatter(
                fam_sub[metric],
                fam_sub["mean_abs_first_to_final_displacement"],
                s=90,
                marker=marker,
                color=FAMILY_COLORS.get(family, "#64748b"),
                edgecolor="white",
                linewidth=0.9,
                zorder=3,
            )

    label_models = _label_extreme_models(data)
    y_span = y_limits[1] - y_limits[0]
    for _, row in data[data["modelo"].isin(label_models)].iterrows():
        dx = 1.05 if row[metric] < np.median(x) else 0.93
        ha = "left" if dx > 1 else "right"
        near_top = row["mean_abs_first_to_final_displacement"] > (
            y_limits[1] - 0.12 * y_span
        )
        dy = -0.030 * y_span if near_top else 0.025 * y_span
        va = "top" if near_top else "bottom"
        ax.text(
            row[metric] * dx,
            row["mean_abs_first_to_final_displacement"] + dy,
            row["model_short"],
            fontsize=8.5,
            ha=ha,
            va=va,
            color="#111827",
        )

    annotation = (
        f"n = {int(stats_row['n'])}\n"
        f"Spearman $\\rho$ = {stats_row['spearman_rho']:.2f}\n"
        f"p = {_format_p_value(stats_row['p_value'])}\n"
        f"95% CI $\\rho$ [{stats_row['ci95_low']:.2f}, {stats_row['ci95_high']:.2f}]"
    )
    ax.text(
        0.03,
        0.03,
        annotation,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cbd5e1"},
    )

    ax.set_xscale("log")
    ax.set_ylim(*y_limits)
    ax.set_title(f"{panel_label}. {title}", loc="left", fontweight="bold")
    ax.set_xlabel(
        "Active parameters per token (B, log scale)"
        if metric == "active_params_b"
        else "Total parameters (B, log scale)"
    )
    ax.grid(True, axis="both", alpha=0.25)


def plot_main_model_size_figure(
    trajectories: pd.DataFrame,
    png_path: Path,
    pdf_path: Path,
    stats_table: pd.DataFrame,
) -> None:
    data = _prepare_size_data(trajectories)
    agg = data[data["scope"] == "Aggregated E1/E2/E3"].copy()
    stats_agg = stats_table[stats_table["scope"] == "Aggregated E1/E2/E3"].set_index("size_metric")

    y = agg["mean_abs_first_to_final_displacement"]
    margin = max(0.04, 0.16 * (y.max() - y.min()))
    y_limits = (max(0, y.min() - margin), y.max() + margin)

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.6), sharey=True)
    _plot_size_panel(
        axes[0],
        agg,
        stats_agg.loc["active_params_b"],
        "active_params_b",
        "Active parameters per token",
        "A",
        y_limits,
    )
    _plot_size_panel(
        axes[1],
        agg,
        stats_agg.loc["total_params_b"],
        "total_params_b",
        "Total parameters",
        "B",
        y_limits,
    )
    axes[0].set_ylabel("Mean absolute first-to-final movement\n(Aggregated E1/E2/E3)")

    family_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color,
            markeredgecolor="white",
            markersize=8,
            label=family,
        )
        for family, color in FAMILY_COLORS.items()
        if family in set(agg["family"])
    ]
    arch_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="#334155",
            linestyle="none",
            markersize=8,
            label=architecture,
        )
        for architecture, marker in ARCHITECTURE_MARKERS.items()
    ]
    trend_handle = Line2D([0], [0], color="#334155", linewidth=2, label="Theil–Sen trend")
    band_handle = Line2D([0], [0], color="#94a3b8", linewidth=8, alpha=0.35, label="95% bootstrap CI")

    leg1 = fig.legend(
        handles=family_handles,
        title="Family",
        loc="lower center",
        bbox_to_anchor=(0.35, -0.02),
        ncol=len(family_handles),
        frameon=False,
    )
    fig.add_artist(leg1)
    fig.legend(
        handles=arch_handles + [trend_handle, band_handle],
        title="Architecture / trend",
        loc="lower center",
        bbox_to_anchor=(0.78, -0.02),
        ncol=2,
        frameon=False,
    )
    # fig.suptitle(
        # "Model size and mean absolute first-to-final movement",
        # fontsize=14,
        # fontweight="bold",
        # y=1.02,
    # )
    fig.tight_layout(rect=[0, 0.10, 1, 0.98])
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def plot_active_size_forest(
    stats_table: pd.DataFrame,
    png_path: Path,
    pdf_path: Path,
) -> None:
    order = [
        "Free summary",
        "E2 — Negative points and positive points",
        "Trade-offs and uncertainty",
        "Aggregated E1/E2/E3",
    ]
    sub = stats_table[
        (stats_table["size_metric"] == "active_params_b")
        & (stats_table["scope"].isin(order))
    ].copy()
    sub["scope"] = pd.Categorical(sub["scope"], categories=order, ordered=True)
    sub = sub.sort_values("scope")

    y_pos = np.arange(len(sub))[::-1]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.axvline(0, color="#475569", linewidth=1.1, linestyle="--")
    xerr = np.vstack(
        [
            sub["spearman_rho"] - sub["ci95_low"],
            sub["ci95_high"] - sub["spearman_rho"],
        ]
    )
    ax.errorbar(
        sub["spearman_rho"],
        y_pos,
        xerr=xerr,
        fmt="o",
        color="#1d4ed8",
        ecolor="#64748b",
        elinewidth=2,
        capsize=4,
        markersize=7,
    )
    for y, (_, row) in zip(y_pos, sub.iterrows()):
        ax.text(
            1.03,
            y,
            f"$\\rho$ = {row['spearman_rho']:.2f} [{row['ci95_low']:.2f}, {row['ci95_high']:.2f}]",
            va="center",
            fontsize=9.5,
        )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sub["scope"])
    ax.set_xlim(-1.05, 1.32)
    ax.set_xlabel("Spearman correlation with active parameters per token")
    ax.set_title(
        "Active model size and movement by condition",
        loc="left",
        fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def plot_leave_one_out(
    leave_one_out: pd.DataFrame,
    png_path: Path,
    pdf_path: Path,
) -> None:
    data = leave_one_out.sort_values("spearman_rho")
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.axhline(0, color="#475569", linewidth=1.1, linestyle="--")
    ax.plot(
        data["removed_model_short"],
        data["spearman_rho"],
        marker="o",
        linewidth=1.8,
        color="#0f766e",
    )
    ax.fill_between(
        np.arange(len(data)),
        data["ci95_low"],
        data["ci95_high"],
        color="#0f766e",
        alpha=0.15,
        linewidth=0,
    )
    ax.set_xticks(np.arange(len(data)))
    ax.set_xticklabels(data["removed_model_short"], rotation=35, ha="right")
    ax.set_ylabel("Leave-one-out Spearman $\\rho$")
    ax.set_title(
        "Sensitivity of aggregated active-size correlation",
        loc="left",
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def create_model_size_figures(
    trajectories: pd.DataFrame,
    root: Path | str = ".",
    save_leave_one_out_figure: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(root)
    png_dir = root / "figures" / "png"
    pdf_dir = root / "figures" / "pdf"
    table_dir = root / "dados" / "analysis_outputs"
    png_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    stats_table, leave_one_out = compute_model_size_statistics(trajectories)
    stats_table.to_csv(table_dir / "model_size_spearman_stats.csv", index=False)
    leave_one_out.to_csv(table_dir / "model_size_leave_one_out.csv", index=False)

    plot_main_model_size_figure(
        trajectories,
        png_dir / "figure_3.png",
        pdf_dir / "figure_3.pdf",
        stats_table,
    )
    plot_active_size_forest(
        stats_table,
        png_dir / "figure_3_forest.png",
        pdf_dir / "figure_3_forest.pdf",
    )
    if save_leave_one_out_figure:
        plot_leave_one_out(
            leave_one_out,
            png_dir / "figure_3_leave_one_out.png",
            pdf_dir / "figure_3_leave_one_out.pdf",
        )

    min_rho = leave_one_out["spearman_rho"].min()
    max_rho = leave_one_out["spearman_rho"].max()
    print(
        "Leave-one-out Spearman rho range "
        f"(active params, aggregated): {min_rho:.3f} to {max_rho:.3f}"
    )
    print(f"Saved model-size statistics to {table_dir / 'model_size_spearman_stats.csv'}")
    print(f"Saved leave-one-out table to {table_dir / 'model_size_leave_one_out.csv'}")
    print(f"Saved figures to {png_dir} and {pdf_dir}")
    return stats_table, leave_one_out


def _load_experiment(root: Path, meta: dict) -> pd.DataFrame:
    df = pd.read_csv(root / meta["path"])
    df["experiment"] = meta["id"]
    df["experiment_name"] = meta["name"]
    df["intervention"] = meta["intervention"]
    df["rodada"] = pd.to_numeric(df.get("rodada", 1), errors="coerce").fillna(1).astype(int)
    df["likert_valor"] = pd.to_numeric(df.get("likert_valor"), errors="coerce")
    tendency_labels = {"esquerda": "Left", "neutro": "Neutral", "direita": "Right"}
    df["condition"] = df["tendencia"].map(tendency_labels).fillna(df["tendencia"])
    df["question_id"] = (
        df["pair_id"].astype(str)
        + "|"
        + df["tipo_pergunta"].astype(str)
        + "|"
        + df["pergunta"].astype(str)
    )
    return df


def build_trajectories_from_csvs(root: Path | str = ".") -> pd.DataFrame:
    root = Path(root)
    responses = pd.concat(
        [_load_experiment(root, meta) for meta in EXPERIMENTS],
        ignore_index=True,
    )
    valid = responses.dropna(subset=["likert_valor"]).copy()
    round_level = (
        valid.groupby(
            ["experiment", "modelo", "condition", "tendencia", "eixo", "question_id", "rodada"],
            dropna=False,
        )["likert_valor"]
        .mean()
        .reset_index()
        .sort_values(["experiment", "modelo", "condition", "eixo", "question_id", "rodada"])
    )
    rows = []
    for keys, group in round_level.groupby(
        ["experiment", "modelo", "condition", "tendencia", "eixo", "question_id"],
        dropna=False,
    ):
        ordered = group.sort_values("rodada")
        values = ordered["likert_valor"].to_numpy(dtype=float)
        if len(np.unique(ordered["rodada"])) <= 1:
            continue
        first = values[0]
        final = values[-1]
        rows.append(
            {
                "experiment": keys[0],
                "modelo": keys[1],
                "condition": keys[2],
                "tendencia": keys[3],
                "eixo": keys[4],
                "question_id": keys[5],
                "first_to_final_abs_delta": abs(final - first),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    project_root = Path.cwd()
    trajectories_df = build_trajectories_from_csvs(project_root)
    create_model_size_figures(trajectories_df, project_root)
