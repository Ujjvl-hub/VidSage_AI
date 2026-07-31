import os

from langchain_groq import ChatGroq

DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"


def get_llm(retries: int = 5):
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set")

    model_name = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)

    llm = ChatGroq(
        model=model_name,
        api_key=api_key,
    )

    # wait_exponential_jitter backs off increasingly between retries instead
    # of hammering an API that just told you it's rate-limited.
    return llm.with_retry(
        stop_after_attempt=retries,
        wait_exponential_jitter=True,
    )