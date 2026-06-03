import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.token_metrics import summarise_token_matrix
from src.visualization import plot_token_comparison_heatmap


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_tokens(token_path: str) -> np.ndarray:
    path = Path(token_path)

    if path.suffix == ".npy":
        return np.load(path)

    if path.suffix == ".csv":
        return np.loadtxt(path, delimiter=",", dtype=int)

    raise ValueError(
        f"Unsupported token file format: {path.suffix}. "
        "Please use .npy or .csv."
    )


def create_demo_tokens(
    num_layers: int = 8,
    human_steps: int = 1100,
    tts_steps: int = 2500,
    vocab_size: int = 1024,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    human = rng.integers(
        low=0,
        high=vocab_size,
        size=(num_layers, human_steps),
    )

    tts = rng.integers(
        low=0,
        high=vocab_size,
        size=(num_layers, tts_steps),
    )

    flat_regions = [
        (580, 660),
        (880, 990),
        (1450, 1580),
        (1850, 1930),
    ]

    for layer in range(num_layers):
        for start, end in flat_regions:
            repeated_token = int(rng.integers(0, vocab_size))
            tts[layer, start:end] = repeated_token

    return human, tts


def save_metrics_csv(rows: list[dict], output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "sample",
        "layer",
        "num_time_steps",
        "num_unique_tokens",
        "entropy_bits",
        "transition_rate",
        "mean_run_length",
        "max_run_length",
    ]

    with open(output, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    config = load_config("configs/exp_token_heatmap.json")

    human_tokens_path = Path(config["human_tokens_path"])
    tts_tokens_path = Path(config["tts_tokens_path"])

    demo_if_missing = bool(config.get("demo_if_missing", False))

    if human_tokens_path.exists() and tts_tokens_path.exists():
        print("[INFO] Loading real acoustic token files...")
        human_tokens = load_tokens(str(human_tokens_path))
        tts_tokens = load_tokens(str(tts_tokens_path))

    elif demo_if_missing:
        print("[INFO] Real token files not found.")
        print("[INFO] Using demo tokens to test the heatmap pipeline.")

        human_tokens, tts_tokens = create_demo_tokens(
            num_layers=8,
            human_steps=1100,
            tts_steps=2500,
            vocab_size=int(config.get("max_token_id", 1023)) + 1,
        )

    else:
        raise FileNotFoundError(
            "Token files were not found.\n"
            f"Expected human tokens at: {human_tokens_path}\n"
            f"Expected TTS tokens at: {tts_tokens_path}\n"
            "Set demo_if_missing=true if you only want to test the pipeline."
        )

    metric_rows = []
    metric_rows.extend(
        summarise_token_matrix(
            tokens=human_tokens,
            sample_name="human_performance_reference",
        )
    )
    metric_rows.extend(
        summarise_token_matrix(
            tokens=tts_tokens,
            sample_name="tts_baseline_generation",
        )
    )

    save_metrics_csv(
        rows=metric_rows,
        output_path=config["output_metrics_path"],
    )

    plot_token_comparison_heatmap(
        human_tokens=human_tokens,
        tts_tokens=tts_tokens,
        output_path=config["output_figure_path"],
        title="Acoustic Token Discrepancy Analysis",
        step_ms=float(config.get("step_ms", 13.33)),
        max_token_id=config.get("max_token_id"),
        colormap=config.get("colormap", "cividis"),
    )

    print("[DONE] Metrics saved to:")
    print(f"  {config['output_metrics_path']}")

    print("[DONE] Figure saved to:")
    print(f"  {config['output_figure_path']}")


if __name__ == "__main__":
    main()