import json
from pathlib import Path


DIALOGUE_PATH = Path("data/text/vw_segment_001_dialogue.json")


def main() -> None:
    with open(DIALOGUE_PATH, "r", encoding="utf-8") as file:
        dialogue = json.load(file)

    for turn in dialogue["turns"]:
        plain_text = turn["plain_text"]

        for key in [
            "punctuation_controlled_text",
            "pause_controlled_text",
            "dramatic_best_effort_text",
        ]:
            value = turn.get(key, "")

            if value.startswith("PASTE_") or value.strip() == "":
                turn[key] = plain_text

    with open(DIALOGUE_PATH, "w", encoding="utf-8") as file:
        json.dump(dialogue, file, indent=2, ensure_ascii=False)

    print("[DONE] Dialogue JSON normalised:")
    print(f"  {DIALOGUE_PATH}")
    print(f"[INFO] Total turns: {len(dialogue['turns'])}")


if __name__ == "__main__":
    main()