from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from core.llm import get_llm


def split_transcript(transcript: str, chunk_size: int = 12000, chunk_overlap: int = 300) -> list:
    """
    chunk_size raised from 3000 -> 12000 chars: Gemini's context window is
    large, so most single-video transcripts now fit in one chunk, which
    means summarize() can skip the map+combine calls entirely below.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(transcript)


def summarize(transcript: str) -> str:
    llm = get_llm()
    chunks = split_transcript(transcript)

    # Short transcript: summarize directly in 1 call instead of
    # map (per chunk) + combine.
    if len(chunks) == 1:
        direct_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert meeting summarizer. Summarize this "
                    "meeting transcript into a professional summary in "
                    "bullet points.",
                ),
                ("human", "{text}"),
            ]
        )
        direct_chain = direct_prompt | llm | StrOutputParser()
        return direct_chain.invoke({"text": chunks[0]})

    map_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Summarize this portion of a meeting transcript concisely."),
            ("human", "{text}"),
        ]
    )
    map_chain = map_prompt | llm | StrOutputParser()
    chunk_summaries = [map_chain.invoke({"text": chunk}) for chunk in chunks]
    combined = "\n\n".join(chunk_summaries)

    combined_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert meeting summarizer. Combine these partial summaries "
                "into one final professional meeting summary in bullet points.",
            ),
            ("human", "{text}"),
        ]
    )

    combined_chain = (
        RunnablePassthrough() | RunnableLambda(lambda x: {"text": x}) | combined_prompt | llm | StrOutputParser()
    )

    return combined_chain.invoke(combined)


def generate_title(transcript: str) -> str:
    llm = get_llm()

    title_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Based on the meeting transcript, generate a short professional meeting title "
                    "(max 8 words). Only return the title, nothing else.",
                ),
                ("human", "{text}"),
            ]
        )
        | llm
        | StrOutputParser()
    )

    return title_chain.invoke(transcript[:2000])