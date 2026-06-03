{
  "experiment_name": "virginia_woolf_moss_generation",
  "corpus_title": "whos_afraid_of_virginia_woolf",
  "segment_id": "vw_segment_001",

  "text": "That was not a very nice thing to say, Martha.",

  "generation_conditions": [
    {
      "condition_id": "plain_baseline",
      "prompt_text": "That was not a very nice thing to say, Martha.",
      "num_takes": 3,
      "output_dir": "data/audio/tts/plain_baseline"
    },
    {
      "condition_id": "pause_controlled",
      "prompt_text": "That was not a very nice thing to say [pause 0.5s] Martha.",
      "num_takes": 3,
      "output_dir": "data/audio/tts/pause_controlled"
    },
    {
      "condition_id": "dramatic_best_effort",
      "prompt_text": "That was NOT... a very nice thing to say [pause 0.5s] Martha.",
      "num_takes": 5,
      "output_dir": "data/audio/tts/dramatic_best_effort"
    }
  ],

  "model_name": "MOSS-TTS-1.5",
  "selection_protocol": {
    "generate_multiple_takes": true,
    "exclude_instruction_leakage": true,
    "exclude_repeated_or_missing_words": true,
    "compare_duration_to_human_reference": true
  }
}