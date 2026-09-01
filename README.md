# STT-AI

STT-AI is a local speech-to-text pipeline for experimenting with audio transcription architecture.

The pipeline reads audio files from `input/`, converts each file to the WAV format expected by the voice activity detection model, identifies regions that contain speech, transcribes those speech regions with Whisper, and writes JSON results to `output/`.

## What it uses

- `ffmpeg` converts source audio into mono, 16 kHz, 16-bit PCM WAV.
- `silero-vad` detects speech frames and speech regions.
- `faster-whisper` transcribes the detected speech regions.
- `torch` runs the VAD and lets the transcription code use CUDA when a compatible GPU is available.

The current transcription model is `small.en`, so the project is configured for English audio.

## Prerequisites

Install these before running the project:

- Python 3.10 or newer
- `ffmpeg`
- Git, if you are cloning the project

On Windows, confirm `ffmpeg` is available from PowerShell:

```powershell
ffmpeg -version
```

If that command is not found, install `ffmpeg` and add it to your `PATH`.

## Setup

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install faster-whisper silero-vad torch torchaudio
```

The first run may download model files for Silero VAD and Faster Whisper. Those model downloads can take a few minutes depending on your network connection.

## Input files

Put audio files in the `input/` folder.

Example:

```text
input/
  recording.m4a
```

The conversion step uses `ffmpeg`, so common audio formats such as `.m4a`, `.mp3`, and `.wav` should work as long as your local `ffmpeg` build supports them.

## Run

From the project root, with the virtual environment activated:

```powershell
python .\src\main.py
```

The script processes every file directly inside `input/`.

## Output

For each input file, the project writes files to `output/`:

- `{name}_stt.wav`: the converted mono 16 kHz WAV file
- `{name}_stt_probabilities.json`: frame-level VAD speech probabilities
- `{name}_stt_transcription.json`: transcription segments with timestamps, word timestamps, and confidence values

The script also prints the generated file paths after each input is processed.

## Troubleshooting

If you see `ffmpeg is not installed or not available in PATH`, install `ffmpeg` or fix your `PATH`.

If model loading or transcription is slow, the project is probably running on CPU. The code uses CUDA automatically when `torch.cuda.is_available()` returns `True`.

If no files are processed, make sure the audio files are directly inside `input/`, not nested inside another folder.
