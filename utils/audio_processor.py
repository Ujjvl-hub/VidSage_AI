import os
import re

import yt_dlp
from pydub import AudioSegment

DOWNLOAD_DIR = "downloads"

os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def _sanitize_filename(name: str) -> str:
    """Strip characters that break file paths on Windows."""
    return re.sub(r'[\\/*?:"<>|]', "", name).strip()


def download_youtube_audio(url: str) -> str:
    """
    Download YouTube audio as WAV.
    """

    output_path = os.path.join(DOWNLOAD_DIR, "%(title).100s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "64",
            }
        ],
        "quiet": True,
        "restrictfilenames": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        filename = os.path.splitext(filename)[0] + ".wav"

    return filename


def convert_to_wav(input_path: str) -> str:
    """
    Convert audio/video to mono 16kHz WAV.
    """

    output_path = os.path.splitext(input_path)[0] + "_16k.wav"

    audio = AudioSegment.from_file(input_path)

    audio = (
        audio
        .set_channels(1)
        .set_frame_rate(16000)
    )

    audio.export(
        output_path,
        format="wav",
        bitrate="64k"
    )

    return output_path


def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list:
    """
    Split WAV into chunks.

    chunk_minutes was 5, sized for the old 4-way parallel transcription.
    Transcription now runs sequentially (see core/transcriber.py), so
    fewer/larger chunks means less per-chunk overhead (export, model call,
    boundary artifacts) without losing anything.
    """

    audio = AudioSegment.from_file(wav_path)

    chunk_ms = chunk_minutes * 60 * 1000

    chunks = []

    total_chunks = (len(audio) + chunk_ms - 1) // chunk_ms

    print(f"Creating {total_chunks} chunk(s)...")

    for i, start in enumerate(range(0, len(audio), chunk_ms)):

        chunk = audio[start:start + chunk_ms]

        chunk_path = f"{wav_path}_chunk_{i}.wav"

        chunk.export(
            chunk_path,
            format="wav",
            parameters=[
                "-ac", "1",
                "-ar", "16000"
            ],
        )

        chunks.append(chunk_path)

    return chunks


def process_input(source: str) -> list:
    """
    Entry point.

    Supports:
    - YouTube URL
    - Local audio/video file
    """

    if source.startswith("http://") or source.startswith("https://"):

        print("Downloading YouTube audio...")

        downloaded_audio = download_youtube_audio(source)

        print("Converting to 16kHz mono...")

        wav_path = convert_to_wav(downloaded_audio)

        # Remove the larger intermediate WAV
        try:
            os.remove(downloaded_audio)
        except Exception:
            pass

    else:

        print("Converting local file to 16kHz mono...")

        wav_path = convert_to_wav(source)

    print("Chunking audio...")

    chunks = chunk_audio(wav_path, chunk_minutes=10)

    print(f"Created {len(chunks)} chunk(s).")

    # wav_path has now been fully split into `chunks` -- the full-length
    # copy is redundant and just takes up disk space, which matters on
    # hosted deployments with limited/ephemeral storage.
    try:
        os.remove(wav_path)
    except Exception:
        pass

    return chunks


def cleanup_chunks(chunks: list) -> None:
    """
    Delete chunk files after they've been transcribed.

    Call this once transcribe_all() has finished with them. Without this,
    every processed video/URL leaves its chunk files behind permanently --
    fine for a one-off local run, but it'll steadily fill up disk on a
    deployed app that gets used repeatedly.
    """
    for path in chunks:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception as e:
            print(f"Could not remove {path}: {e}")