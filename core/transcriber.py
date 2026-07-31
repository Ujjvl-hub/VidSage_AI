import os

import requests
from faster_whisper import WhisperModel
from pydub import AudioSegment

SARVAM_PIECE_SECONDS = 25

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "tiny")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")

_model = None


def load_model():
    global _model

    if _model is None:
        print(f"Loading Whisper model: {WHISPER_MODEL}...")

        # NOTE: os.cpu_count() reports the HOST's core count, not what's
        # actually allotted to this container (e.g. Streamlit Community
        # Cloud free tier ~= 1 usable core). Requesting more threads than
        # you actually have causes contention/thrashing, not speed-up.
        cpu_threads = int(os.getenv("WHISPER_CPU_THREADS", "1"))

        _model = WhisperModel(
            WHISPER_MODEL,
            device="cpu",
            compute_type=WHISPER_COMPUTE_TYPE,
            cpu_threads=cpu_threads,
            num_workers=1,
        )

        print("Whisper model loaded.")

    return _model


def transcribe_chunk_whisper(chunk_path: str) -> str:
    model = load_model()

    segments, _ = model.transcribe(
        chunk_path,
        task="transcribe",
        beam_size=1,
        vad_filter=True,
        word_timestamps=False,
    )

    return " ".join(segment.text.strip() for segment in segments)


def _send_to_sarvam(piece_path: str) -> str:
    headers = {"api-subscription-key": SARVAM_API_KEY}

    with open(piece_path, "rb") as f:
        files = {"file": (os.path.basename(piece_path), f, "audio/wav")}
        data = {
            "model": SARVAM_MODEL,
            "with_diarization": "false",
        }

        response = requests.post(
            SARVAM_STT_TRANSLATE_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=120,
        )

    if not response.ok:
        print(response.text)
        response.raise_for_status()

    return response.json().get("transcript", "")


def transcribe_chunk_sarvam(chunk_path: str) -> str:
    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY not found.")

    audio = AudioSegment.from_wav(chunk_path)

    piece_ms = SARVAM_PIECE_SECONDS * 1000

    full_text = ""

    total_pieces = (len(audio) + piece_ms - 1) // piece_ms

    for i, start in enumerate(range(0, len(audio), piece_ms)):
        piece = audio[start:start + piece_ms]

        piece_path = f"{chunk_path}_sv_{i}.wav"

        piece.export(piece_path, format="wav")

        try:
            print(f"Sarvam piece {i+1}/{total_pieces}")

            full_text += _send_to_sarvam(piece_path) + " "

        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)

    return full_text.strip()


def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    if language.lower() == "hinglish":
        return transcribe_chunk_sarvam(chunk_path)

    return transcribe_chunk_whisper(chunk_path)


def transcribe_all(chunks: list, language: str = "english") -> str:

    engine = "Sarvam AI" if language.lower() == "hinglish" else "Whisper"

    print(f"Using {engine} on {len(chunks)} chunk(s)")

    transcripts = []
    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i + 1}/{len(chunks)}...")
        transcripts.append(transcribe_chunk(chunk, language))
        print(f"Chunk {i + 1}/{len(chunks)} done.")

    print("Transcription complete.")

    return " ".join(transcripts).strip()