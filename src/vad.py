from pathlib import Path
import wave

import torch

SAMPLE_RATE = 16_000  
WINDOW_SIZE = 512 

SPEECH_THRESHOLD = 0.50 # Threshold for detecting speech in the audio
MAX_SILENCE_MS = 300
MIN_SPEECH_MS = 250
SPEECH_PADDING_MS = 100
MERGE_GAP_MS = 500 # Maximum gap between speech regions to merge them
MAX_SEGMENT_MS =  30000 # Maximum length of a speech segment in milliseconds

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

def frames_to_speech_regions(frames, audio_length):
    regions = []

    min_speech_samples = int(SAMPLE_RATE * MIN_SPEECH_MS / 1000) # Calculate the minimum number of samples required for a speech region based on the minimum speech duration in milliseconds
    max_silence_samples = int(SAMPLE_RATE * MAX_SILENCE_MS / 1000) # Calculate the maximum number of samples allowed for silence between speech regions based on the maximum silence duration in milliseconds
    padding_samples = int(SAMPLE_RATE * SPEECH_PADDING_MS / 1000) # Calculate the number of samples to pad around speech regions based on the speech padding duration in milliseconds

    region_start = None
    last_speech_end = None
    speech_probabilities = []

    for frame in frames:
        if frame["is_speech"]:
            # Start a new region on the first speech frame
            if region_start is None:
                region_start = frame["start"]
                speech_probabilities = []

            last_speech_end = frame["end"] # Update the end of the last speech frame to the current frame's end
            speech_probabilities.append(frame["probability"]) 
            continue

        # No active speech region
        if region_start is None:
            continue

        # Keep the region open during short pauses
        silence_length = frame["end"] - last_speech_end
        if silence_length <= max_silence_samples:
            continue

        # Enough silence has passed, so close the region at end of the last actual speech frame
        speech_length = last_speech_end - region_start

        if speech_length >= min_speech_samples:
            regions.append({
                "start_sample": max(0, region_start - padding_samples),
                "end_sample": min(audio_length, last_speech_end + padding_samples),
                "average_probability": sum(speech_probabilities) / len(speech_probabilities)
            },
        )

        region_start = None
        last_speech_end = None
        speech_probabilities = []

    # File might end while speech is still active, so check if we need to close the last region
    if region_start is not None:
        speech_length = last_speech_end - region_start
        if speech_length >= min_speech_samples: # If the last speech region is long enough, add it to the list of regions
            regions.append({
                "start_sample": max(0, region_start - padding_samples),
                "end_sample": min(audio_length, last_speech_end + padding_samples),
                "average_probability": sum(speech_probabilities) / len(speech_probabilities)
            })

    return regions

def merge_speech_regions(regions):
    if not regions:
        return []

    merge_gap_samples = int(
        SAMPLE_RATE * MERGE_GAP_MS / 1000
    )

    max_segment_samples = int( 
        SAMPLE_RATE * MAX_SEGMENT_MS / 1000
    )

    merged = [regions[0].copy()] # Start with the first region as the initial merged region

    # [1:] to skip the first region since it's already added to merged
    for next_region in regions[1:]:
        current_region = merged[-1] # Get the last region in the merged list to compare with the next region

        gap = (
            next_region["start_sample"] - current_region["end_sample"] # Calculate the gap between the current region and the next region
        )

        combined_length = (
            next_region["end_sample"] - current_region["start_sample"] # Calculate the combined length of the current region and the next region
        )

        close_enough = gap <= merge_gap_samples # Check if the gap between the two regions is small enough to consider merging them
        within_max_length = combined_length <= max_segment_samples # Check if the combined length of the two regions is within the maximum allowed segment length

        if close_enough and within_max_length:
            current_duration = (
                current_region["end_sample"] - current_region["start_sample"] # Calculate the duration of the current region
            )
            next_duration = (
                next_region["end_sample"] - next_region["start_sample"] # Calculate the duration of the next region
            )

            total_duration = current_duration + next_duration # Calculate the total duration of the merged region

            current_region["average_probability"] = (
            (
                current_region["average_probability"] * current_duration
            ) + (
                next_region["average_probability"] * next_duration
            )
            ) / total_duration # Update the average probability of the merged region based on the weighted average of the two regions

            current_region["end_sample"] = max(
                current_region["end_sample"], next_region["end_sample"],
            )
        else:
            merged.append(next_region.copy()) # If the regions cannot be merged, add the next region as a new entry in the merged list
            current_region = next_region
    return merged