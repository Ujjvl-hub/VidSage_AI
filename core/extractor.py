import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.llm import get_llm


# ---------------------------------------------------------------------------
# Chain builder
# ---------------------------------------------------------------------------

def build_chain(system_prompt: str):
    llm = get_llm()
    return (
        RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{text}"),
        ])
        | llm
        | StrOutputParser()
    )


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_transcript(
    transcript: str,
    chunk_size: int = 15000,
    chunk_overlap: int = 500,
) -> list:
    """
    Split a long transcript into overlapping chunks.

    Gemini has a very large context window, so chunk_size is set high on
    purpose -- the goal is to minimize LLM calls (and therefore API quota
    usage), only splitting when the transcript genuinely doesn't fit.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(transcript)


# ---------------------------------------------------------------------------
# Combined extraction (action items + decisions + questions in ONE pass)
#
# Previously these were 3 separate map-reduce chains -- up to 15 LLM calls
# for a 4-chunk transcript. Combining them into a single pass with clearly
# labeled sections cuts that to ~5 calls (or 1, for a short transcript).
# ---------------------------------------------------------------------------

SECTION_HEADERS = ("Action Items", "Key Decisions", "Open Questions")

COMBINED_CHUNK_PROMPT = (
    "You are an expert meeting analyst. From this excerpt of a meeting "
    "transcript, extract three things found in THIS EXCERPT ONLY:\n\n"
    "## Action Items\n"
    "Numbered list. Each item: Task, Owner (if mentioned, else 'Not "
    "specified'), Deadline (if mentioned, else 'Not specified'). If none, "
    "write exactly: NONE\n\n"
    "## Key Decisions\n"
    "Numbered list of key decisions made. If none, write exactly: NONE\n\n"
    "## Open Questions\n"
    "Numbered list of unresolved questions or topics needing follow-up. "
    "If none, write exactly: NONE\n\n"
    "Respond using exactly these three '## ' headers, in this order, and "
    "nothing else."
)

COMBINED_REDUCE_PROMPT = (
    "You are given partial Action Items / Key Decisions / Open Questions, "
    "each using the same three '## ' headers, extracted from consecutive "
    "chunks of the same meeting transcript. Merge them into one final "
    "result using the same three headers in the same order. Deduplicate "
    "items that clearly refer to the same thing. If a section ends up "
    "empty, write one sentence saying nothing was found for it. Respond "
    "using exactly the three headers and nothing else."
)


def _split_sections(text: str) -> dict:
    """Parse a '## Header' formatted response into {header: body}."""
    pattern = r"##\s*(" + "|".join(re.escape(h) for h in SECTION_HEADERS) + r")\s*\n"
    parts = re.split(pattern, text)

    result = {h: "" for h in SECTION_HEADERS}
    # re.split with a capturing group -> [pre, header, body, header, body, ...]
    for i in range(1, len(parts) - 1, 2):
        header = parts[i].strip()
        body = parts[i + 1].strip()
        if header in result:
            result[header] = body

    return result


_EMPTY_MESSAGES = {
    "Action Items": "No action items found.",
    "Key Decisions": "No key decisions found.",
    "Open Questions": "No open questions found.",
}


def _clean_section(header: str, body: str) -> str:
    body = body.strip()
    if not body or body.upper() == "NONE":
        return _EMPTY_MESSAGES[header]
    return body


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_meeting(transcript: str, chunk_size: int = 15000, chunk_overlap: int = 500) -> dict:
    """
    Single entry point for meeting analysis. Extracts action items, key
    decisions, and open questions using as few LLM calls as possible:
    1 call for a short transcript, or N map calls + 1 reduce call for a
    long one -- instead of doing that 3x over for each category separately.
    """
    chunks = chunk_transcript(transcript, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    map_chain = build_chain(COMBINED_CHUNK_PROMPT)

    if len(chunks) == 1:
        sections = _split_sections(map_chain.invoke(chunks[0]))
    else:
        partial_results = [map_chain.invoke(chunk) for chunk in chunks]
        reduce_chain = build_chain(COMBINED_REDUCE_PROMPT)
        combined_input = "\n\n---\n\n".join(partial_results)
        sections = _split_sections(reduce_chain.invoke(combined_input))

    return {
        "action_items": _clean_section("Action Items", sections["Action Items"]),
        "key_decisions": _clean_section("Key Decisions", sections["Key Decisions"]),
        "open_questions": _clean_section("Open Questions", sections["Open Questions"]),
    }