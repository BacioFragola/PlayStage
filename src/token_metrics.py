import numpy as np

def normalize_token_matrix(tokens: np.ndarray) -> np.ndarray:
    """
    Standardise token matrix into shape:
        layers x time_steps

    Acoustic token files may be save in two common formats:
        1. layers x time steps
        2. time_steps x layers

    This function makes sure later analysis always uses:
        layers x time_steps
    """
    matrix = np.asarray(tokens)

    if matrix.ndim != 2:
        raise ValueError(
            f"Expected a 2D token matrix, but got shape {matrix.shape}."
        )
    
    if matrix.shape[0] > matrix.shape[1] and matrix.shape[1] <= 64:
        matrix = matrix.T

    return matrix.astype(int)

def count_unique_tokens(sequence: np.ndarray) -> int:
    """
    Count how many different token IDs appear in one codebook layer.

    Research meaning:
        More unique tokens may indicate richer acoustic variation.
    """
    return int(len(np.unique(sequence)))

def token_entropy(sequence: np.ndarray) -> float:
    """
    Calculate Shannon entropy over token IDs.

    Research meaning:
        Higher entropy may suggests more diverse and less concentrated token usage.
        Lower entropy may suggests more repetitive or concentrated token usage.
    """
    if len(sequence) == 0:
        return 0.0
    
    _, counts = np.unique(sequence, return_counts=True)
    probabilities = counts / counts.sum()

    entropy = -np.sum(probabilities * np.log2(probabilities))
    return float(entropy)

def transition_rate(sequence: np.ndarray) -> float:
    """
    Calculate how often adjacent token IDs change.

    Research meaning:
        Higher transition rates may indicate more dynamic token movement.
        Lower transition rates may indicate flatter or more stable token behaviour.
    """
    if len(sequence) <= 1:
        return 0.0
    
    transitions = np.diff(sequence) != 0
    return float(np.mean(transitions))

def run_lengths(sequence: np.ndarray) -> list[int]:
    """
    Return the lengths of consecutive repeated-token runs.
    """
    if len(sequence) == 0:
        return []
    
    lengths = []
    current_length = 1

    for i in range(1, len(sequence)):
        if sequence[i] == sequence[i - 1]:
            current_length += 1
        else:
            lengths.append(current_length)
            current_length = 1
    
    lengths.append(current_length)
    return lengths

def average_run_length(sequence: np.ndarray) -> float:
    """
    Calculate the average length of consecutive repeated-token runs.

    Research meaning:
        Longer average run lengths may indicate more repetitive or flatter token behaviour.
    """
    lengths = run_lengths(sequence)
    if not lengths:
        return 0.0
    
    return float(np.mean(lengths))

def max_run_length(sequence: np.ndarray) -> int:
    """
    Calculate the longest repeated-token runs.

    Research meaning:
        A very long max run lengths may indicate a sustained flat or repetive region.
    """
    lengths = run_lengths(sequence)

    if not lengths:
        return 0
    
    return int(np.max(lengths))

def summarise_token_layer(sequence: np.ndarray, sample_name: str, layer_index: int) -> dict:
    """
    Calculate all token metrics for one codebook layer.
    """
    return {
        "sample": sample_name,
        "layer": f"L{layer_index + 1}",
        "num_time_steps": int(len(sequence)),
        "num_unique_tokens": count_unique_tokens(sequence),
        "entropy_bits": round(token_entropy(sequence), 4),
        "transition_rate": round(transition_rate(sequence), 4),
        "mean_run_length": round(average_run_length(sequence), 4),
        "max_run_length": max_run_length(sequence),
    }

def summarise_token_matrix(tokens: np.ndarray, sample_name: str) -> list[dict]:
    """
    Calculate layer-wise token metrics for a complete acoustic token matrix. 

    Input:
        tokens: acoustic token matrix

    Expected internal format:
        layers x time_steps

    Output:
        A list of dictionaries.
        Each dictionary is one codebook layer's metrics.
    """
    matrix = normalize_token_matrix(tokens)
    rows = []

    for layer_index, sequence in enumerate(matrix):
        row = summarise_token_layer(
            sequence=sequence,
            sample_name=sample_name,
            layer_index=layer_index,
        )
        rows.append(row)

    return rows