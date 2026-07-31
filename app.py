"""
AI Video Assistant - Streamlit UI
Turns a YouTube link or uploaded audio/video file into a searchable,
chat-able meeting brief: title, summary, action items, decisions,
open questions, and a RAG chatbot grounded in the transcript.
"""

import os
import time
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from utils.audio_processor import process_input, cleanup_chunks
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import analyze_meeting
from core.RAG_Engine import build_rag_chain, ask_questions


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        .block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1200px; }

        /* Hero header */
        .hero {
            background: linear-gradient(120deg, #4338CA 0%, #6D28D9 45%, #DB2777 100%);
            border-radius: 20px;
            padding: 2.4rem 2.6rem;
            margin-bottom: 1.8rem;
            box-shadow: 0 10px 30px rgba(76, 29, 149, 0.25);
        }
        .hero h1 {
            color: #ffffff;
            font-size: 2.1rem;
            font-weight: 800;
            margin-bottom: 0.35rem;
            letter-spacing: -0.02em;
        }
        .hero p {
            color: rgba(255,255,255,0.88);
            font-size: 1.02rem;
            margin: 0;
        }

        /* Section cards */
        .card {
            background: var(--background-color, #ffffff);
            border: 1px solid rgba(120,120,140,0.15);
            border-radius: 16px;
            padding: 1.4rem 1.6rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.04);
        }

        /* Metric pills */
        .pill {
            display: inline-block;
            padding: 0.3rem 0.85rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            margin-right: 0.4rem;
        }
        .pill-green  { background: #DCFCE7; color: #166534; }
        .pill-blue   { background: #DBEAFE; color: #1E40AF; }
        .pill-amber  { background: #FEF3C7; color: #92400E; }
        .pill-purple { background: #EDE9FE; color: #5B21B6; }

        /* Buttons */
        div.stButton > button {
            border-radius: 10px;
            font-weight: 600;
            padding: 0.55rem 1.2rem;
            border: none;
        }
        div.stButton > button[kind="primary"] {
            background: linear-gradient(120deg, #4338CA, #DB2777);
        }

        section[data-testid="stSidebar"] {
            border-right: 1px solid rgba(120,120,140,0.15);
        }

        .footer-note {
            text-align: center;
            color: #888;
            font-size: 0.8rem;
            margin-top: 2.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
for key, default in {
    "result": None,
    "chat_history": [],
    "processing": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🎬 VidSage AI</h1>
        <p>Understand every video with AI-generated transcripts, insights, and contextual conversations.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar - inputs & system status
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    source_type = st.radio("Source", ["YouTube URL", "Upload file"], horizontal=True)

    source = None
    if source_type == "YouTube URL":
        source = st.text_input("YouTube URL", placeholder="https://youtu.be/...")
    else:
        uploaded = st.file_uploader(
            "Upload audio or video",
            type=["mp3", "wav", "m4a", "mp4", "mov", "mkv"],
        )
        if uploaded is not None:
            os.makedirs("uploads", exist_ok=True)
            save_path = os.path.join("uploads", uploaded.name)
            with open(save_path, "wb") as f:
                f.write(uploaded.getbuffer())
            source = save_path
            st.success(f"Saved: {uploaded.name}")

    language = st.selectbox(
        "Language",
        ["english", "hinglish"],
        help="english → local Whisper model. hinglish → Sarvam AI (Hindi/English mix → English).",
    )

    st.markdown("---")
    process_clicked = st.button(
        "🚀 Process", type="primary", use_container_width=True, disabled=not source
    )

    st.markdown("---")
    st.markdown("### 🔑 System status")
    groq_ok = bool(os.getenv("GROQ_API_KEY"))
    sarvam_ok = bool(os.getenv("SARVAM_API_KEY"))
    st.markdown(
        f"{'🟢' if groq_ok else '🔴'} Groq API key &nbsp;&nbsp; "
        f"{'🟢' if sarvam_ok else '⚪'} Sarvam API key (hinglish only)",
        unsafe_allow_html=True,
    )
    # if not groq_ok:
    #     st.caption("Set `GROQ_API_KEY` in your `.env` file to enable summarization & chat.")

    st.markdown("---")
    st.caption("Built with LangChain · Faster-Whisper · Groq · ChromaDB · Streamlit")

# ---------------------------------------------------------------------------
# Pipeline execution with live progress
# ---------------------------------------------------------------------------
if process_clicked and source:
    st.session_state.chat_history = []
    start_time = time.time()

    with st.status("Running pipeline...", expanded=True) as status:
        try:
            status.update(label="🎧 Extracting & chunking audio...")
            chunks = process_input(source)

            status.update(label=f"📝 Transcribing {len(chunks)} chunk(s)...")
            transcript = transcribe_all(chunks, language=language)
            cleanup_chunks(chunks)

            status.update(label="🏷️ Generating title...")
            title = generate_title(transcript)

            status.update(label="📋 Summarizing transcript...")
            summary = summarize(transcript)

            status.update(label="✅ Extracting action items, decisions & questions...")
            analysis = analyze_meeting(transcript)

            status.update(label="🧠 Building knowledge base for chat...")
            rag_chain = build_rag_chain(transcript)

            elapsed = time.time() - start_time
            status.update(label=f"Done in {elapsed:.1f}s", state="complete", expanded=False)

            st.session_state.result = {
                "title": title,
                "transcript": transcript,
                "summary": summary,
                "action_items": analysis["action_items"],
                "key_decisions": analysis["key_decisions"],
                "open_questions": analysis["open_questions"],
                "rag_chain": rag_chain,
                "n_chunks": len(chunks),
                "processed_at": datetime.now().strftime("%b %d, %Y · %I:%M %p"),
            }

        except Exception as e:
            status.update(label="Pipeline failed", state="error", expanded=True)
            st.error(f"Something went wrong: {e}")

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
result = st.session_state.result

if result is None:
    st.info("👈 Add a YouTube URL or upload a file in the sidebar, then hit **Process** to get started.")
else:
    st.markdown(f"## 📌 {result['title']}")
    word_count = len(result["transcript"].split())
    st.markdown(
        f"""
        <span class="pill pill-blue">{result['n_chunks']} audio chunk(s)</span>
        <span class="pill pill-purple">{word_count:,} words transcribed</span>
        <span class="pill pill-green">Processed {result['processed_at']}</span>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")

    tabs = st.tabs(
        ["📋 Summary", "📝 Transcript", "✅ Action Items", "🔑 Decisions", "❓ Questions", "💬 Chat"]
    )

    with tabs[0]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(result["summary"])
        st.markdown("</div>", unsafe_allow_html=True)
        st.download_button(
            "⬇️ Download summary (.txt)",
            data=result["summary"],
            file_name="summary.txt",
            use_container_width=False,
        )

    with tabs[1]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write(result["transcript"])
        st.markdown("</div>", unsafe_allow_html=True)
        st.download_button(
            "⬇️ Download transcript (.txt)",
            data=result["transcript"],
            file_name="transcript.txt",
        )

    with tabs[2]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(result["action_items"])
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[3]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(result["key_decisions"])
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[4]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(result["open_questions"])
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[5]:
        st.caption("Ask anything about the video — answers are grounded only in the transcript.")

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        question = st.chat_input("Ask a question about this video...")
        if question:
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = ask_questions(result["rag_chain"], question)
                    st.markdown(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

st.markdown(
    '<p class="footer-note">AI Video Assistant · LangChain + Groq + Faster-Whisper + ChromaDB</p>',
    unsafe_allow_html=True,
)