import argparse
import csv
import importlib.util
import os
import sys
import wave
from pathlib import Path


# ------------------------------------------------------------
# 1. Fix Windows / conda DLL loading before importing torch
# ------------------------------------------------------------

CONDA_ENV = Path(r"C:\ProgramData\anaconda3\envs\moss-tts")

for dll_dir in [
    CONDA_ENV,
    CONDA_ENV / "DLLs",
    CONDA_ENV / "Library" / "bin",
    CONDA_ENV / "Library" / "usr" / "bin",
]:
    if dll_dir.exists():
        os.add_dll_directory(str(dll_dir))
        os.environ["PATH"] = f"{dll_dir};" + os.environ.get("PATH", "")


import numpy as np
import torch


# ------------------------------------------------------------
# 2. Project paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MOSS_APP_PATH = Path(
    r"C:\Users\22014868\Documents\OpenMOSS\MOSS-TTS\clis\moss_tts_app.py"
)

MOSS_MODEL_PATH = Path(
    r"C:\Users\22014868\.cache\huggingface\hub\models--OpenMOSS-Team--MOSS-TTS-v1.5\snapshots\cdd3b911b1585e3f2dbc7775ef10f9926f58850a"
)

DEFAULT_QUEUE_PATH = PROJECT_ROOT / "outputs/tables/virginia_woolf_moss_generation_queue.csv"


# ------------------------------------------------------------
# 3. Utility functions
# ------------------------------------------------------------

def load_moss_app():
    if not MOSS_APP_PATH.exists():
        raise FileNotFoundError(f"MOSS app file not found: {MOSS_APP_PATH}")

    spec = importlib.util.spec_from_file_location("moss_tts_app", MOSS_APP_PATH)

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load MOSS app from: {MOSS_APP_PATH}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["moss_tts_app"] = module
    spec.loader.exec_module(module)

    return module


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def load_queue(queue_path: Path) -> list[dict]:
    if not queue_path.exists():
        raise FileNotFoundError(f"Queue file not found: {queue_path}")

    with open(queue_path, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader)


def patch_torchaudio_load_with_wave() -> None:
    """
    Patch torchaudio.load so MOSS-TTS can read Audacity-exported
    16-bit PCM WAV reference audio without relying on TorchCodec.
    """
    import torchaudio

    def load_audio_with_wave(filepath, *args, **kwargs):
        with wave.open(str(filepath), "rb") as wav_file:
            num_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            num_frames = wav_file.getnframes()
            raw_audio = wav_file.readframes(num_frames)

        if sample_width != 2:
            raise ValueError(
                f"Expected 16-bit PCM WAV, but got sample width: {sample_width} bytes"
            )

        audio = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32)
        audio = audio / 32768.0

        if num_channels > 1:
            audio = audio.reshape(-1, num_channels).T
        else:
            audio = audio.reshape(1, -1)

        audio_tensor = torch.from_numpy(audio.copy())

        return audio_tensor, sample_rate

    torchaudio.load = load_audio_with_wave


def describe_tensor(name: str, value) -> None:
    if isinstance(value, torch.Tensor):
        print(f"{name}:")
        print(f"  type: torch.Tensor")
        print(f"  shape: {tuple(value.shape)}")
        print(f"  dtype: {value.dtype}")
        print(f"  device: {value.device}")
    else:
        print(f"{name}:")
        print(f"  type: {type(value)}")
        if hasattr(value, "shape"):
            print(f"  shape: {value.shape}")


def describe_batch(batch: dict) -> None:
    print("\n" + "=" * 80)
    print("STAGE 3 — Processor output batch")
    print("=" * 80)

    print(f"batch keys: {list(batch.keys())}")

    for key, value in batch.items():
        describe_tensor(f"batch['{key}']", value)


# ------------------------------------------------------------
# 4. Main debug pipeline
# ------------------------------------------------------------

def debug_one_job(job: dict, args: argparse.Namespace) -> None:
    print("\n" + "=" * 80)
    print("STAGE 1 — Raw job from queue")
    print("=" * 80)

    reference_audio_path = resolve_project_path(job["reference_audio_path"])

    print(f"condition_id: {job['condition_id']}")
    print(f"turn_id: {job['turn_id']}")
    print(f"speaker: {job['speaker']}")
    print(f"text: {job['text']}")
    print(f"performance_note: {job['performance_note']}")
    print(f"reference_audio_path: {reference_audio_path}")
    print(f"reference exists: {reference_audio_path.exists()}")

    if not reference_audio_path.exists():
        raise FileNotFoundError(f"Reference audio not found: {reference_audio_path}")

    print("\n" + "=" * 80)
    print("STAGE 2 — Load MOSS backend")
    print("=" * 80)

    patch_torchaudio_load_with_wave()
    print("[INFO] Patched torchaudio.load with Python wave reader.")

    moss_app = load_moss_app()

    model, processor, torch_device, sample_rate = moss_app.load_backend(
        model_path=str(MOSS_MODEL_PATH),
        device_str=args.device,
        attn_implementation=moss_app.DEFAULT_ATTN_IMPLEMENTATION,
    )

    print(f"model type: {type(model)}")
    print(f"processor type: {type(processor)}")
    print(f"torch_device: {torch_device}")
    print(f"sample_rate: {sample_rate}")

    if hasattr(model, "config"):
        print("\nModel config fields that may be useful:")
        for attr in [
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "vocab_size",
            "max_position_embeddings",
        ]:
            if hasattr(model.config, attr):
                print(f"  {attr}: {getattr(model.config, attr)}")

    print("\n" + "=" * 80)
    print("STAGE 3 — Build conversation and processor batch")
    print("=" * 80)

    conversations, mode, mode_name = moss_app.build_conversation(
        text=job["text"],
        reference_audio=str(reference_audio_path),
        mode_with_reference=moss_app.MODE_CLONE,
        expected_tokens=None,
        language_tag="English",
        processor=processor,
    )

    print(f"mode: {mode}")
    print(f"mode_name: {mode_name}")
    print(f"conversation type: {type(conversations)}")
    print(f"number of conversations: {len(conversations)}")

    batch = processor(conversations, mode=mode)
    describe_batch(batch)

    input_ids = batch["input_ids"].to(torch_device)
    attention_mask = batch["attention_mask"].to(torch_device)

    print("\n" + "=" * 80)
    print("STAGE 3B — Inspect input_ids channels")
    print("=" * 80)

    input_cpu = input_ids.detach().cpu()

    print("input_ids full shape:", tuple(input_cpu.shape))
    print("First 5 sequence positions, all 33 channels:")
    print(input_cpu[0, :5, :])

    print("\nChannel-wise statistics:")
    for channel_index in range(input_cpu.shape[-1]):
        channel_values = input_cpu[0, :, channel_index]
        unique_count = torch.unique(channel_values).numel()
        min_value = channel_values.min().item()
        max_value = channel_values.max().item()

        print(
        f"channel {channel_index:02d}: "
        f"min={min_value}, max={max_value}, unique={unique_count}"
    )
        
    print("\nPositions where audio channels are active:")

    audio_channels = input_cpu[0, :, 1:]

    # We assume 1024 is the empty / placeholder token in audio channels.
    audio_active = (audio_channels != 1024).any(dim=1)

    active_indices = torch.where(audio_active)[0]

    print("number of active audio positions:", active_indices.numel())

    if active_indices.numel() > 0:
        print("first active audio position:", active_indices[0].item())
        print("last active audio position:", active_indices[-1].item())

        print("\nFirst 5 active audio rows:")
        for idx in active_indices[:5]:
            print(f"position {idx.item()}:")
            print(input_cpu[0, idx, :])

    print("\nKey model inputs after moving to device:")
    describe_tensor("input_ids", input_ids)
    describe_tensor("attention_mask", attention_mask)

    print("\nInterpretation:")
    print("  input_ids shape [B, S]")
    print("  B = batch size")
    print("  S = total model input sequence length")
    print("  S may include text tokens, special tokens, and reference-audio-related tokens.")

    print("\n" + "=" * 80)
    print("STAGE 4 — Autoregressive generation")
    print("=" * 80)

    print("Calling model.generate(...). This may take some time.")

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=int(args.max_new_tokens),
            audio_temperature=float(args.temperature),
            audio_top_p=float(args.top_p),
            audio_top_k=int(args.top_k),
            audio_repetition_penalty=float(args.repetition_penalty),
        )

    describe_tensor("outputs", outputs)

    print("\nInterpretation:")
    print("  outputs contains generated token/code sequences.")
    print("  In an autoregressive model, generation extends the original sequence")
    print("  by predicting new audio-related tokens step by step.")

    print("\n" + "=" * 80)
    print("STAGE 5 — Decode generated audio")
    print("=" * 80)

    messages = processor.decode(outputs)

    print(f"messages type: {type(messages)}")
    print(f"number of decoded messages: {len(messages)}")

    if not messages or messages[0] is None:
        raise RuntimeError("No decoded message returned by processor.decode(outputs).")

    print(f"first message type: {type(messages[0])}")

    audio = messages[0].audio_codes_list[0]

    describe_tensor("decoded audio", audio)

    if isinstance(audio, torch.Tensor):
        audio_np = audio.detach().float().cpu().numpy()
    else:
        audio_np = np.asarray(audio, dtype=np.float32)

    print("\nFinal waveform:")
    print(f"  audio_np shape: {audio_np.shape}")
    print(f"  audio_np dtype: {audio_np.dtype}")
    print(f"  sample_rate: {sample_rate}")
    print(f"  duration_sec: {audio_np.reshape(-1).shape[0] / sample_rate:.3f}")

    print("\n" + "=" * 80)
    print("SUMMARY — Shape lifecycle")
    print("=" * 80)

    print("Raw text + reference wav")
    print("  ↓")
    print("processor(...)")
    print(f"  input_ids: {tuple(input_ids.shape)}")
    print(f"  attention_mask: {tuple(attention_mask.shape)}")
    print("  ↓")
    print("model.generate(...)")
    if hasattr(outputs, "shape"):
        print(f"  outputs: {tuple(outputs.shape)}")
    else:
        print(f"  outputs type: {type(outputs)}")
    print("  ↓")
    print("processor.decode(...)")
    print(f"  waveform: {audio_np.shape}")
    print("  ↓")
    print("wav audio can be saved by the generation script.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue_path", type=str, default=str(DEFAULT_QUEUE_PATH))
    parser.add_argument("--job_index", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--max_new_tokens", type=int, default=512)

    args = parser.parse_args()

    queue_path = Path(args.queue_path)
    jobs = load_queue(queue_path)

    if args.job_index < 1 or args.job_index > len(jobs):
        raise ValueError(
            f"job_index must be between 1 and {len(jobs)}, got {args.job_index}"
        )

    job = jobs[args.job_index - 1]

    debug_one_job(job=job, args=args)


if __name__ == "__main__":
    main()