# 🎬 AI Video & Meeting Assistant

Turns any YouTube video or local recording into a transcript, summary, extracted
action items/decisions/questions, and a RAG chatbot grounded in that transcript.

## Features
- **Audio pipeline**: `yt-dlp` (YouTube) or direct upload → `pydub`/`ffmpeg` conversion → chunking
- **Transcription**: local `faster-whisper` (English) or Sarvam AI (Hindi→English translation)
- **LLM analysis**: Gemini via LangChain (LCEL) — title, map-reduce summary, action items, decisions, open questions
- **RAG chat**: Chroma vector store + HuggingFace embeddings, so you can ask follow-up questions about the video
- **UI**: Streamlit, with live pipeline progress, tabs, and downloadable transcript/summary

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

Install [ffmpeg](https://ffmpeg.org/download.html) and make sure it's on your PATH (required by `pydub`/`yt-dlp`).

Copy `.env.example` to `.env` and add your `GOOGLE_API_KEY` (get one at
[Google AI Studio](https://aistudio.google.com/app/apikey)).

## Run

**Streamlit UI (recommended):**
```bash
streamlit run app.py
```

**CLI:**
```bash
python main.py
```

## Project structure
```
AI_VideoAssistant_Project/
├── app.py                  # Streamlit UI
├── main.py                 # CLI entry point
├── core/
│   ├── transcriber.py       # faster-whisper / Sarvam transcription
│   ├── summarize.py         # title + map-reduce summary
│   ├── extractor.py         # action items / decisions / questions
│   ├── RAG_Engine.py        # chat-with-transcript chain
│   └── vector_stores.py     # Chroma vector store
├── utils/
│   └── audio_processor.py   # download/convert/chunk audio
└── requirements.txt
```

## Notes
- If you previously hit `ImportError: DLL load failed while importing _internal`
  from `numba` — this project uses `faster-whisper` (CTranslate2 backend) instead
  of `openai-whisper`, which avoids that dependency entirely.
- Avoid running the project from inside a OneDrive-synced folder; sync locks and
  security scanning can interfere with virtual environments.
