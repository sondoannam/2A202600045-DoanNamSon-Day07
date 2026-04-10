from __future__ import annotations


import os
from pathlib import Path
from dotenv import load_dotenv

from src.agent import KnowledgeBaseAgent
from src.chunking import CustomChunker
from src.embeddings import OpenAIEmbedder, _mock_embed
from src.models import Document
from src.store import EmbeddingStore

# ===== CONFIG =====
DATA_DIR = "data"
DB_PATH = "data/vector_db.pkl"

# ===== LOAD ALL FILES =====
def load_all_markdown(data_dir: str) -> list[Document]:
    docs = []
    path = Path(data_dir)

    for file in path.glob("*.md"):
        content = file.read_text(encoding="utf-8")

        docs.append(
            Document(
                id=file.stem,
                content=content,
                metadata={
                    "source": str(file),
                    "filename": file.name
                },
            )
        )

    return docs


# ===== BUILD VECTOR DB =====
def build_vectordb(embedder):
    print("🔄 Building vector DB from multiple files...")


    raw_docs = load_all_markdown(DATA_DIR)
    chunker = CustomChunker(chunk_size=200)


    all_chunks = []


    for doc in raw_docs:
        chunks = chunker.chunk(doc.content)


        for i, chunk in enumerate(chunks):
            all_chunks.append(
                Document(
                    id=f"{doc.id}_{i}",
                    content=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_id": i
                    },
                )
            )


    print(f"📄 Total chunks: {len(all_chunks)}")


    store = EmbeddingStore(embedding_fn=embedder)
    store.add_documents(all_chunks)


    os.makedirs("docs", exist_ok=True)
    store.save(DB_PATH)


    print(f"✅ Vector DB saved at: {DB_PATH}")


# ===== LOAD DB =====
def load_vectordb(embedder):
    store = EmbeddingStore(embedding_fn=embedder)
    store.load(DB_PATH)
    return store


# ===== LLM =====
def demo_llm(prompt: str) -> str:
    return "📌 Answer:\n" + prompt[-400:]


# ===== CHAT =====
def chat():
    load_dotenv()

    try:
        embedder = OpenAIEmbedder()
        print("🔗 Using OpenAI Embedding")
    except Exception:
        embedder = _mock_embed
        print("⚠️ Using mock embedding")


    if not os.path.exists(DB_PATH):
        build_vectordb(embedder)


    store = load_vectordb(embedder)
    agent = KnowledgeBaseAgent(store, demo_llm)


    print("\n💬 Multi-doc RAG Chatbot Ready (type 'exit' to quit)\n")


    while True:
        q = input("You: ")


        if q.lower() == "exit":
            break


        results = store.search(q, top_k=3)


        print("\n🔍 Top chunks:")
        for r in results:
            print(f"- {r['metadata'].get('filename')} (chunk {r['metadata'].get('chunk_id')})")


        answer = agent.answer(q)


        print("\n💡 Answer:")
        print(answer)
        print("-" * 50)




# ===== MAIN =====
if __name__ == "__main__":
    chat()