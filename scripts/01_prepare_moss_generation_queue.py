import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_json(path: str) -> dict:
    """
    Load a JSON file as a Python dictionary.

    Research role:
        This allows the generation experiment to be controlled by external
        config and dialogue files instead of hard-coded values.
    """
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def get_character_map(characters: list[dict]) -> dict:
    """
    Convert the character list into a dictionary indexed by character_id.

    Example:
        [
            {"character_id": "george", ...},
            {"character_id": "martha", ...}
        ]

    becomes:
        {
            "george": {...},
            "martha": {...}
        }
    """
    character_map = {}

    for character in characters:
        character_id = character["character_id"]
        character_map[character_id] = character

    return character_map


def build_output_audio_path(
    output_dir: str,
    segment_id: str,
    condition_id: str,
    turn_id: str,
    speaker: str,
    take_number: int,
) -> str:
    """
    Build a clear output filename for each generated audio file.

    Example:
        vw_segment_001_george_martha_confrontation_plain_baseline_t001_george_take_01.wav
    """
    filename = (
        f"{segment_id}_{condition_id}_{turn_id}_{speaker}_"
        f"take_{take_number:02d}.wav"
    )

    return str(Path(output_dir) / filename)


def build_generation_queue(config: dict, dialogue: dict) -> list[dict]:
    """
    Build a generation queue from:
        1. experiment config
        2. dialogue script

    Each row represents one MOSS-TTS generation job.

    Research role:
        This makes the generation process auditable:
        each output audio file can be traced back to a speaker, turn,
        condition, text variant, reference voice, and take number.
    """
    character_map = get_character_map(config["characters"])
    segment_id = config["segment_id"]

    queue_rows = []

    for condition in config["generation_conditions"]:
        condition_id = condition["condition_id"]
        text_variant = condition["text_variant"]
        num_takes = int(condition["num_takes"])
        output_dir = condition["output_dir"]

        for turn in dialogue["turns"]:
            turn_id = turn["turn_id"]
            speaker = turn["speaker"]

            if speaker not in character_map:
                raise ValueError(
                    f"Speaker '{speaker}' in dialogue is not defined in config characters."
                )

            if text_variant not in turn:
                raise ValueError(
                    f"Text variant '{text_variant}' is missing in turn '{turn_id}'."
                )

            character = character_map[speaker]
            text = turn[text_variant]

            for take_number in range(1, num_takes + 1):
                output_audio_path = build_output_audio_path(
                    output_dir=output_dir,
                    segment_id=segment_id,
                    condition_id=condition_id,
                    turn_id=turn_id,
                    speaker=speaker,
                    take_number=take_number,
                )

                queue_rows.append(
                    {
                        "experiment_name": config["experiment_name"],
                        "segment_id": segment_id,
                        "condition_id": condition_id,
                        "turn_id": turn_id,
                        "speaker": speaker,
                        "display_name": character.get("display_name", speaker),
                        "voice_profile": character.get("voice_profile", ""),
                        "reference_audio_path": character.get("reference_audio_path", ""),
                        "text_variant": text_variant,
                        "text": text,
                        "performance_note": turn.get("performance_note", ""),
                        "take_number": take_number,
                        "output_audio_path": output_audio_path,
                    }
                )

    return queue_rows


def save_queue_csv(rows: list[dict], output_path: str) -> None:
    """
    Save the generation queue as a CSV table.

    This CSV is the formal generation plan for MOSS-TTS.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "experiment_name",
        "segment_id",
        "condition_id",
        "turn_id",
        "speaker",
        "display_name",
        "voice_profile",
        "reference_audio_path",
        "text_variant",
        "text",
        "performance_note",
        "take_number",
        "output_audio_path",
    ]

    with open(output, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    config = load_json("configs/exp_moss_generation.json")
    dialogue = load_json(config["dialogue_script_path"])

    queue_rows = build_generation_queue(
        config=config,
        dialogue=dialogue,
    )

    output_path = config["outputs"]["generation_queue_path"]

    save_queue_csv(
        rows=queue_rows,
        output_path=output_path,
    )

    print("[DONE] MOSS generation queue saved to:")
    print(f"  {output_path}")
    print(f"[INFO] Total generation jobs: {len(queue_rows)}")


if __name__ == "__main__":
    main()