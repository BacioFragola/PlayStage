from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.token_metrics import normalize_token_matrix


def plot_token_comparison_heatmap(
    human_tokens: np.ndarray,
    tts_tokens: np.ndarray,
    output_path: str,
    title: str = "Acoustic Token Discrepancy Analysis",
    human_label: str = "Human Performance Reference (Who's Afraid of Virginia Woolf?)",
    tts_label: str = "TTS Baseline Generation",
    step_ms: float = 13.33,
    max_token_id: int | None = None,
    colormap: str = "cividis",
) -> None:
    """
    Plot two acoustic token heatmaps for comparison.

    Panel A:
        Human performance reference.

    Panel B:
        TTS-generated speech.

    Rows:
        Codebook layers.

    Columns:
        Time steps.

    Cell colour:
        Audio token ID.

    Research purpose:
        This visualisation helps inspect token-level temporal structure,
        repetition, flatness, and dynamic variation across codebook layers.
    """
    human = normalize_token_matrix(human_tokens)
    tts = normalize_token_matrix(tts_tokens)

    if human.shape[0] != tts.shape[0]:
        raise ValueError(
            "Human and TTS token matrices must have the same number of layers. "
            f"Got human={human.shape}, tts={tts.shape}."
        )

    num_layers = human.shape[0]

    if max_token_id is None:
        vmax = int(max(np.max(human), np.max(tts)))
    else:
        vmax = int(max_token_id)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(14, 9),
    )

    fig.suptitle(
        title,
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )

    panels = [
        (f"(a) {human_label}", human),
        (f"(b) {tts_label}", tts),
    ]

    image = None

    for ax, (panel_title, matrix) in zip(axes, panels):
        image = ax.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
            cmap=colormap,
            vmin=0,
            vmax=vmax,
        )

        ax.set_title(
            panel_title,
            loc="left",
            fontsize=13,
            fontweight="bold",
        )

        ax.set_ylabel("Codebook Layers", fontsize=12)
        ax.set_yticks(np.arange(num_layers))
        ax.set_yticklabels([f"L{i + 1}" for i in range(num_layers)])

        num_steps = matrix.shape[1]
        tick_positions = np.linspace(0, num_steps - 1, 6, dtype=int)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([str(position) for position in tick_positions])

    axes[-1].set_xlabel(
        f"Time Steps ({step_ms:.2f}ms per step)",
        fontsize=12,
    )

    colorbar = fig.colorbar(
        image,
        ax=axes,
        fraction=0.035,
        pad=0.04,
    )
    colorbar.set_label("Audio Token ID", fontsize=12)

    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)