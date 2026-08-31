import json
from pathlib import Path

from silero_vad import load_silero_vad

from convert_audio import convert_audio
from vad import calculate_probabilities, classify_frames, merge_speech_regions, read_wav, frames_to_speech_regions


PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_DIR / "input"
OUTPUT_DIR = PROJECT_DIR / "output"


def main():
    input_files = sorted(path for path in INPUT_DIR.iterdir() if path.is_file())
    if not input_files:
        print(f"No files found in {INPUT_DIR}")
        return

    model = load_silero_vad()

    for input_path in input_files:
        print(f"Processing {input_path.name}...")
        wav_path = convert_audio(input_path, OUTPUT_DIR)
        audio = read_wav(wav_path) # Read the WAV file into a normalized torch tensor
        frames = calculate_probabilities(audio, model) # Calculate the probability of speech for each frame in the audio
        frames = classify_frames(frames) # Classify each frame as speech or non-speech based on the probability threshold
        frames = frames_to_speech_regions(frames, len(audio)) # Convert classified frames into speech regions with start and end samples
        frames = merge_speech_regions(frames)  # Merge speech regions based on the defined gap

        probabilities_path = wav_path.with_name(
            f"{wav_path.stem}_probabilities.json"
        )
        probabilities_path.write_text(
            json.dumps(frames, indent=2), encoding="utf-8"
        )

        print(f"  WAV: {wav_path}")
        print(f"  VAD probabilities: {probabilities_path} ({len(frames)} frames)")

if __name__ == "__main__":
    main()
