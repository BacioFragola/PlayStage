import argparse
import csv
import re
import wave
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_QUEUE_PATH = PROJECT_ROOT / "outputs/tables/virginia_woolf_moss_generation_queue.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data/audio/tts/stitched/vw_segment_001_moss_plain_baseline_full.wav"
DEFAULT_STITCH_METADATA_PATH = PROJECT_ROOT / "outputs/tables/virginia_woolf_stitched_dialogue_metadata.csv"


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


def turn_sort_key(turn_id: str) -> int:
    matched = re.search(r"\d+", turn_id)

    if matched is None:
        return 0

    return int(matched.group())


def read_wav_bytes(wav_path: Path) -> tuple[bytes, int, int, int, int]:
    if not wav_path.exists():
        raise FileNotFoundError(f"Missing generated wav file: {wav_path}")

    with wave.open(str(wav_path), "rb") as wav_file:
        num_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        num_frames = wav_file.getnframes()
        audio_bytes = wav_file.readframes(num_frames)

    return audio_bytes, num_channels, sample_width, sample_rate, num_frames


def make_silence_bytes(
    duration_seconds: float,
    num_channels: int,
    sample_width: int,
    sample_rate: int,
) -> bytes:
    silence_frames = int(duration_seconds * sample_rate)
    bytes_per_frame = num_channels * sample_width

    return b"\x00" * silence_frames * bytes_per_frame


def save_stitch_metadata(rows: list[dict], metadata_path: Path) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "order",
        "turn_id",
        "speaker",
        "condition_id",
        "input_audio_path",
        "duration_sec",
        "text",
    ]

    with open(metadata_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def stitch_dialogue(
    queue_path: Path,
    condition_id: str,
    take_number: int,
    silence_seconds: float,
    output_path: Path,
    metadata_path: Path,
) -> None:
    rows = load_queue(queue_path)

    selected_rows = [
        row
        for row in rows
        if row["condition_id"] == condition_id
        and int(row["take_number"]) == take_number
    ]

    if not selected_rows:
        raise ValueError(
            f"No rows found for condition_id={condition_id}, take_number={take_number}"
        )

    selected_rows = sorted(
        selected_rows,
        key=lambda row: turn_sort_key(row["turn_id"]),
    )

    stitched_audio = bytearray()
    stitch_metadata = []

    expected_channels = None
    expected_sample_width = None
    expected_sample_rate = None

    for index, row in enumerate(selected_rows, start=1):
        wav_path = resolve_project_path(row["output_audio_path"])

        (
            audio_bytes,
            num_channels,
            sample_width,
            sample_rate,
            num_frames,
        ) = read_wav_bytes(wav_path)

        if expected_channels is None:
            expected_channels = num_channels
            expected_sample_width = sample_width
            expected_sample_rate = sample_rate

        if (
            num_channels != expected_channels
            or sample_width != expected_sample_width
            or sample_rate != expected_sample_rate
        ):
            raise ValueError(
                "WAV format mismatch:\n"
                f"  file: {wav_path}\n"
                f"  got: channels={num_channels}, sample_width={sample_width}, sample_rate={sample_rate}\n"
                f"  expected: channels={expected_channels}, sample_width={expected_sample_width}, sample_rate={expected_sample_rate}"
            )

        duration_sec = num_frames / sample_rate

        stitched_audio.extend(audio_bytes)

        if index < len(selected_rows):
            stitched_audio.extend(
                make_silence_bytes(
                    duration_seconds=silence_seconds,
                    num_channels=num_channels,
                    sample_width=sample_width,
                    sample_rate=sample_rate,
                )
            )

        stitch_metadata.append(
            {
                "order": index,
                "turn_id": row["turn_id"],
                "speaker": row["speaker"],
                "condition_id": row["condition_id"],
                "input_audio_path": str(wav_path),
                "duration_sec": round(duration_sec, 3),
                "text": row["text"],
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(output_path), "wb") as output_file:
        output_file.setnchannels(expected_channels)
        output_file.setsampwidth(expected_sample_width)
        output_file.setframerate(expected_sample_rate)
        output_file.writeframes(bytes(stitched_audio))

    save_stitch_metadata(stitch_metadata, metadata_path)

    total_duration = len(stitched_audio) / (
        expected_sample_rate * expected_channels * expected_sample_width
    )

    print("[DONE] Stitched dialogue saved to:")
    print(f"  {output_path}")
    print(f"[INFO] Total turns stitched: {len(selected_rows)}")
    print(f"[INFO] Silence between turns: {silence_seconds:.2f}s")
    print(f"[INFO] Output duration: {total_duration:.2f}s")
    print("[DONE] Stitch metadata saved to:")
    print(f"  {metadata_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue_path", type=str, default=str(DEFAULT_QUEUE_PATH))
    parser.add_argument("--condition_id", type=str, default="plain_baseline")
    parser.add_argument("--take_number", type=int, default=1)
    parser.add_argument("--silence_seconds", type=float, default=0.35)
    parser.add_argument("--output_path", type=str, default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--metadata_path", type=str, default=str(DEFAULT_STITCH_METADATA_PATH))
    args = parser.parse_args()

    stitch_dialogue(
        queue_path=Path(args.queue_path),
        condition_id=args.condition_id,
        take_number=args.take_number,
        silence_seconds=args.silence_seconds,
        output_path=Path(args.output_path),
        metadata_path=Path(args.metadata_path),
    )


if __name__ == "__main__":
    main()