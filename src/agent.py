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
        context_parts = []
        for res in results:
            text_to_use = res['metadata'].get('parent_content', res['content'])
            context_parts.append(f"- {text_to_use}")
            
        context = "\n\n".join(context_parts)
        prompt = (
            "Bạn là trợ lý ảo hỗ trợ chính sách Shopee. Hãy trả lời câu hỏi dựa TRÊN NGỮ CẢNH ĐƯỢC CUNG CẤP DƯỚI ĐÂY.\n"
            "Nếu ngữ cảnh không chứa thông tin, hãy nói 'Tôi không tìm thấy thông tin trong tài liệu'.\n\n"
            f"NGỮ CẢNH:\n{context}\n\n"
            f"CÂU HỎI: {question}\n"
            "TRẢ LỜI:"
        )
        
        # Call LLM
        return self.llm_fn(prompt)
