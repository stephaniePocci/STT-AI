from pathlib import Path
import subprocess


def convert_audio(input_path: Path, output_dir: Path) -> Path:
    """Convert an audio file to the mono, 16 kHz WAV expected by Silero VAD."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}_stt.wav"
    command = [
        "ffmpeg", "-y", "-i", str(input_path), "-vn",
        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        str(output_path),
    ]

    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as error:
        raise RuntimeError("ffmpeg is not installed or not available in PATH") from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"ffmpeg could not convert {input_path.name}") from error

    return output_path
