from pathlib import Path
import wave

import torch

SAMPLE_RATE = 16_000  
WINDOW_SIZE = 512 

SPEECH_THRESHOLD = 0.50 # Threshold for detecting speech in the audio
MAX_SILENCE_MS = 300
MIN_SPEECH_MS = 250
SPEECH_PADDING_MS = 100


def read_wav(path: Path):
    """Read a mono, 16-bit PCM WAV file into a normalized torch tensor."""
    with wave.open(str(path), "rb") as wav_file:
        if wav_file.getnchannels() != 1:
            raise ValueError("Expected mono audio")
        if wav_file.getsampwidth() != 2:
            raise ValueError("Expected 16-bit PCM audio")
        if wav_file.getframerate() != SAMPLE_RATE:
            raise ValueError(f"Expected a {SAMPLE_RATE} Hz sample rate")
        samples = wav_file.readframes(wav_file.getnframes())

    return torch.frombuffer(bytearray(samples), dtype=torch.int16).float() / 32768.0

def calculate_probabilities(audio, model):
    frames = []

    model.reset_states()

    for start_sample in range(0, len(audio), WINDOW_SIZE):
        end_sample = min(start_sample + WINDOW_SIZE, len(audio))
        chunk = audio[start_sample:end_sample] # Get the current chunk of audio samples

        if len(chunk) < WINDOW_SIZE:
            missing_samples = WINDOW_SIZE - len(chunk) # Calculate how many samples are missing to reach the window size
            chunk = torch.nn.functional.pad(chunk, (0, missing_samples)) # Pad the chunk with zeros to ensure it has the correct size

        probability = model(chunk, SAMPLE_RATE).item() # Get the probability of speech in the chunk
        frames.append({
            "start": start_sample,
            "end": end_sample,
            "probability": probability,
        })

    model.reset_states() # Reset the model's internal states after processing all chunks
    return frames

def classify_frames(frames):
    classified_frames = [] # classify each frame as speech or non-speech based on the probability threshold

    for frame in frames:
        classified_frames.append(
            {
                **frame,
                "is_speech": frame["probability"] >= SPEECH_THRESHOLD
            }
        )
    return classified_frames
