from dotenv import load_dotenv

load_dotenv()  # must run before core/ modules read env vars

from utils.audio_processor import process_input, cleanup_chunks
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import analyze_meeting
from core.RAG_Engine import build_rag_chain, ask_questions


def run_pipeline(source: str, language: str = "english") -> dict:
    print("Starting AI Video Assistant...")

    chunks = process_input(source)
    transcript = transcribe_all(chunks, language=language)
    cleanup_chunks(chunks)

    print(f"Raw transcription (first 300 characters):\n{transcript[:300]}")

    title = generate_title(transcript)
    summary = summarize(transcript)
    analysis = analyze_meeting(transcript)

    rag_chain = build_rag_chain(transcript)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": analysis["action_items"],
        "key_decisions": analysis["key_decisions"],
        "open_questions": analysis["open_questions"],
        "rag_chain": rag_chain,
    }


if __name__ == "__main__":
    source = input("Enter YouTube URL or local file path: ").strip()
    language = input("Language (english/hinglish): ").strip() or "english"

    result = run_pipeline(source, language)

    print("\n" + "=" * 60)
    print(f"Title: {result['title']}")
    print(f"\nSummary:\n{result['summary']}")
    print(f"\nAction Items:\n{result['action_items']}")
    print(f"\nKey Decisions:\n{result['key_decisions']}")
    print(f"\nOpen Questions:\n{result['open_questions']}")
    print("=" * 60)

    print("\nChat with your meeting (type 'exit' to quit)\n")

    rag_chain = result["rag_chain"]

    while True:
        user_question = input("You: ").strip()

        if user_question.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break

        if not user_question:
            continue

        answer = ask_questions(rag_chain, user_question)
        print(f"\nAssistant: {answer}\n")