import json
from pathlib import Path

from silero_vad import load_silero_vad

from convert_audio import convert_audio
from vad import calculate_probabilities, classify_frames, merge_speech_regions, read_wav, frames_to_speech_regions
from transcription import transcribe_region, transcription_model


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
        regions = frames_to_speech_regions(frames, len(audio)) # Convert classified frames into speech regions with start and end samples
        regions = merge_speech_regions(regions)  # Merge speech regions based on the defined gap

        transcription = []

        for region in regions:
            transcription.extend(
                transcribe_region(audio, region, transcription_model)
            )

        results_path = wav_path.with_name(
            f"{wav_path.stem}_transcription.json"
        )

        results_path.write_text(
            json.dumps(
                {
                    "source": input_path.name,
                    "regions": regions,
                    "transcription": transcription,
                    "text": " ".join(segment["text"] for segment in transcription),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        probabilities_path = wav_path.with_name(
            f"{wav_path.stem}_probabilities.json"
        )
        probabilities_path.write_text(
            json.dumps(frames, indent=2), encoding="utf-8"
        )

        transcription_path = wav_path.with_name(
            f"{wav_path.stem}_transcription.json"
        )
        transcription_path.write_text(
            json.dumps(transcription, indent=2), encoding="utf-8"
        )

        print(f"  WAV: {wav_path}")
        print(f"  VAD probabilities: {probabilities_path} ({len(frames)} frames)")
        print(f"  Transcription: {transcription_path}")

if __name__ == "__main__":
    main()
