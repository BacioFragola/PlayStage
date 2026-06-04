import argparse
import csv
import importlib.util
import os
import sys
import time
import traceback
import wave
from pathlib import Path


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


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MOSS_APP_PATH = Path(
    r"C:\Users\22014868\Documents\OpenMOSS\MOSS-TTS\clis\moss_tts_app.py"
)

DEFAULT_QUEUE_PATH = PROJECT_ROOT / "outputs/tables/virginia_woolf_moss_generation_queue.csv"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "outputs/tables/virginia_woolf_moss_generation_metadata.csv"


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

def patch_torchaudio_load_with_wave() -> None:
    """
    Patch torchaudio.load so MOSS-TTS can read WAV reference audio
    without TorchCodec on Windows.

    This only supports standard PCM WAV files, which is fine for
    our Audacity-exported reference voices.
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


def save_wav(output_path: Path, sample_rate: int, audio: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    audio = np.clip(audio, -1.0, 1.0)

    audio_int16 = (audio * 32767.0).astype(np.int16)

    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(audio_int16.tobytes())


def save_metadata(rows: list[dict], metadata_path: Path) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "job_index",
        "condition_id",
        "turn_id",
        "speaker",
        "text",
        "reference_audio_path",
        "output_audio_path",
        "status",
        "elapsed_sec",
        "error",
    ]

    with open(metadata_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_one_job(
    moss_app,
    job: dict,
    job_index: int,
    args: argparse.Namespace,
) -> dict:
    reference_audio_path = resolve_project_path(job["reference_audio_path"])
    output_audio_path = resolve_project_path(job["output_audio_path"])

    if not reference_audio_path.exists():
        raise FileNotFoundError(
            f"Reference audio not found for speaker '{job['speaker']}':\n"
            f"  {reference_audio_path}"
        )

    if output_audio_path.exists() and args.skip_existing:
        return {
            "job_index": job_index,
            "condition_id": job["condition_id"],
            "turn_id": job["turn_id"],
            "speaker": job["speaker"],
            "text": job["text"],
            "reference_audio_path": str(reference_audio_path),
            "output_audio_path": str(output_audio_path),
            "status": "skipped_existing",
            "elapsed_sec": 0.0,
            "error": "",
        }

    started_at = time.monotonic()

    audio_result, moss_status = moss_app.run_inference(
        text=job["text"],
        reference_audio=str(reference_audio_path),
        mode_with_reference=moss_app.MODE_CLONE,
        duration_control_enabled=False,
        duration_tokens=0,
        language_tag="English",
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=int(args.top_k),
        repetition_penalty=float(args.repetition_penalty),
        model_path=r"C:\Users\22014868\.cache\huggingface\hub\models--OpenMOSS-Team--MOSS-TTS-v1.5\snapshots\cdd3b911b1585e3f2dbc7775ef10f9926f58850a",
        device=args.device,
        attn_implementation=moss_app.DEFAULT_ATTN_IMPLEMENTATION,
        max_new_tokens=int(args.max_new_tokens),
    )

    sample_rate, audio_np = audio_result
    save_wav(output_audio_path, sample_rate, audio_np)

    elapsed = time.monotonic() - started_at

    return {
        "job_index": job_index,
        "condition_id": job["condition_id"],
        "turn_id": job["turn_id"],
        "speaker": job["speaker"],
        "text": job["text"],
        "reference_audio_path": str(reference_audio_path),
        "output_audio_path": str(output_audio_path),
        "status": moss_status,
        "elapsed_sec": round(elapsed, 3),
        "error": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue_path", type=str, default=str(DEFAULT_QUEUE_PATH))
    parser.add_argument("--metadata_path", type=str, default=str(DEFAULT_METADATA_PATH))
    parser.add_argument("--max_jobs", type=int, default=3)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    queue_path = Path(args.queue_path)
    metadata_path = Path(args.metadata_path)

    jobs = load_queue(queue_path)

    selected_jobs = jobs[args.start_index : args.start_index + args.max_jobs]

    print(f"[INFO] Loaded queue: {queue_path}")
    print(f"[INFO] Total jobs in queue: {len(jobs)}")
    print(f"[INFO] Running jobs: {len(selected_jobs)}")
    print(f"[INFO] Loading MOSS-TTS app from: {MOSS_APP_PATH}")

    patch_torchaudio_load_with_wave()
    print("[INFO] Patched torchaudio.load to use Python wave reader.")

    moss_app = load_moss_app()

    metadata_rows = []

    for local_index, job in enumerate(selected_jobs, start=1):
        job_index = args.start_index + local_index

        print("=" * 80)
        print(f"[JOB {job_index}/{len(jobs)}]")
        print(f"speaker: {job['speaker']}")
        print(f"turn_id: {job['turn_id']}")
        print(f"text: {job['text']}")
        print(f"reference: {job['reference_audio_path']}")
        print(f"output: {job['output_audio_path']}")

        try:
            row = generate_one_job(
                moss_app=moss_app,
                job=job,
                job_index=job_index,
                args=args,
            )
            print("[DONE] Generated:")
            print(f"  {row['output_audio_path']}")
            print(f"  elapsed: {row['elapsed_sec']}s")

        except Exception as error:
            row = {
                "job_index": job_index,
                "condition_id": job.get("condition_id", ""),
                "turn_id": job.get("turn_id", ""),
                "speaker": job.get("speaker", ""),
                "text": job.get("text", ""),
                "reference_audio_path": job.get("reference_audio_path", ""),
                "output_audio_path": job.get("output_audio_path", ""),
                "status": "failed",
                "elapsed_sec": 0.0,
                "error": repr(error),
            }

            print("[ERROR] Job failed:")
            print(traceback.format_exc())

        metadata_rows.append(row)

    save_metadata(metadata_rows, metadata_path)

    print("=" * 80)
    print("[DONE] Generation metadata saved to:")
    print(f"  {metadata_path}")


if __name__ == "__main__":
    main()