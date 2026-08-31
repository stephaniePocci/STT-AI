import torch
from faster_whisper import WhisperModel

from vad import SAMPLE_RATE

transcription_model = WhisperModel(
    "small.en",
    device="cuda" if torch.cuda.is_available() else "cpu",
    compute_type="int8",
)

def transcribe_region(audio, region, model):
    start_sample = region["start_sample"]
    end_sample = region["end_sample"]

    if end_sample <= start_sample:
        raise ValueError(f"Invalid region: start_sample ({start_sample}) must be less than end_sample ({end_sample})")

    audio_chunk = audio[start_sample:end_sample] # Get the audio samples for the specified region

    audio_chunk = (
        audio_chunk.detach()
        .cpu()
        .numpy()
        .astype("float32", copy=False)
    ) # Convert the audio chunk to a NumPy array of type float32

    whisper_segments, info = model.transcribe(
        audio_chunk,
        language="en",
        word_timestamps=True, # Request word-level timestamps in the transcription output
        vad_filter=False, # Apply voice activity detection to filter out non-speech segments from the audio
        condition_on_previous_text=True,
        beam_size=5, # Controls the number of beams used in beam search decoding. A higher value may improve accuracy but will increase computation time.
        best_of=5, # Determines how many candidate transcriptions are generated for each segment.
    )

    # Timestamps returned by Whisper are relative to audio_chunk, so we need to adjust them to be relative to the original audio
    region_offset = start_sample / SAMPLE_RATE # Calculate the offset of the region in seconds
    results = []

    for segment in whisper_segments:
        words = []
        word_confidences = []

        for word in segment.words or []:
            confidence = word.probability
            word_confidences.append(confidence)

            words.append({
                "word": word.word,
                "start": region_offset + word.start, # Adjust the start time of the word to be relative to the original audio
                "end": region_offset + word.end, # Adjust the end time of the word to be relative to the original audio
                "confidence": confidence
            })

        average_confidence = (
            sum(word_confidences) / len(word_confidences)
            if word_confidences
            else None
        )

        results.append({
            "start": region_offset + segment.start, # Adjust the start time of the segment to be relative to the original audio
            "end": region_offset + segment.end, # Adjust the end time of the segment to be relative to the original audio
            "text": segment.text.strip(), # Remove leading and trailing whitespace from the segment text
            "words": words, # Include the list of words with their adjusted timestamps
            "confidence": average_confidence
        })

    return results
