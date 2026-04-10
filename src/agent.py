from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        # Retrieve top-k relevant chunks from the store
        results = self.store.search(question, top_k)
        
        # Build prompt
        context = "\n\n".join([f"Chunk {i+1}:\n{chunk['content']}" for i, chunk in enumerate(results)])
        prompt = f"You are a helpful assistant. Use the context below to answer the question.If the answer is not in the context, say you don't know.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
        
        # 3. Call LLM
        return self.llm_fn(prompt)
