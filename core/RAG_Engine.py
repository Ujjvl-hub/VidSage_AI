from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from core.llm import get_llm
from core.vector_stores import (
    build_vector_store,
    load_vector_store,
    get_retriever,
)


# ----------------------------------------------------------------------
# Utility
# ----------------------------------------------------------------------

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# ----------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------

RAG_SYSTEM_PROMPT = """
You are an intelligent AI Video Assistant.

You have access to a meeting/video transcript.

Your responsibilities:

- Answer questions using the transcript whenever it contains relevant information.
- If the transcript partially answers the question, combine it with your own knowledge.
- Clearly mention which information comes from the transcript.
- Never invent transcript details.
- If the transcript does not contain the answer, say that the transcript doesn't mention it and answer using your own knowledge.
- Be concise, conversational and accurate.

Transcript:
{context}
"""


GENERAL_SYSTEM_PROMPT = """
You are an intelligent AI assistant.

Answer the user's question naturally using your own knowledge.

Be conversational, accurate and concise.
"""


# ----------------------------------------------------------------------
# Build Chains
# ----------------------------------------------------------------------

def build_rag_chain_internal(retriever, llm):

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RAG_SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )

    return (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )


def build_general_chain(llm):

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", GENERAL_SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )

    return (
        prompt
        | llm
        | StrOutputParser()
    )


# ----------------------------------------------------------------------
# Hybrid RAG
# ----------------------------------------------------------------------

class HybridRAG:

    def __init__(self, vector_store):

        self.vector_store = vector_store

        self.llm = get_llm()

        self.retriever = get_retriever(
            vector_store,
            k=4,
        )

        self.rag_chain = build_rag_chain_internal(
            self.retriever,
            self.llm,
        )

        self.general_chain = build_general_chain(
            self.llm,
        )

    def ask(self, question: str):

        # Retrieve documents with similarity scores
        results = self.vector_store.similarity_search_with_relevance_scores(
            question,
            k=4,
        )

        # No retrieved documents
        if not results:
            return self.general_chain.invoke(
                {
                    "question": question,
                }
            )

        docs = [doc for doc, score in results]

        best_score = results[0][1]

        print(f"Best relevance score: {best_score:.3f}")

        # Threshold can be tuned later
        THRESHOLD = 0.45

        # Low relevance -> general AI
        if best_score < THRESHOLD:

            return self.general_chain.invoke(
                {
                    "question": question,
                }
            )

        # High relevance -> transcript RAG
        return self.rag_chain.invoke(question)


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def build_rag_chain(transcript: str):

    vector_store = build_vector_store(transcript)

    return HybridRAG(vector_store)


def load_rag_chain():

    vector_store = load_vector_store()

    return HybridRAG(vector_store)


def ask_questions(rag_chain, question: str):

    return rag_chain.ask(question)