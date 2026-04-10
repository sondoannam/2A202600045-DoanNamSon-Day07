import os
from openai import OpenAI
from dotenv import load_dotenv
from src.chunking import ParentChildChunker
from src.models import Document
from src.store import EmbeddingStore
from src.embeddings import OpenAIEmbedder
from src.agent import KnowledgeBaseAgent
import time

def run_benchmark():
    load_dotenv()
    # 1. Setup Data
    files = [
        "data/shopee_chinh_sach_tra_hang_hoan_tien.md", 
        "data/shopee_dong_kiem.md",
        "data/shopee_huy_don_hoan_voucher.md"
    ]
    
    chunker = ParentChildChunker(max_sentences_per_child=2)
    docs = []
    
    print("Đang băm tài liệu...")
    for filename in files:
        if not os.path.exists(filename):
            print(f"Báo lỗi: Không tìm thấy {filename}")
            continue
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read()
        chunks = chunker.chunk_with_metadata(text)
        for i, c in enumerate(chunks):
            # Tạo ID duy nhất bằng cách ghép tên file và index
            doc_id = f"{os.path.basename(filename)}_{i}"
            docs.append(Document(id=doc_id, content=c["content"], metadata=c["metadata"]))

    # 2. Setup Vector Store & Embedder (Dùng OpenAI)
    print("Đang tạo Vector Embeddings (Tốn vài giây)...")
    embedder = OpenAIEmbedder(model_name="text-embedding-3-small")
    # Đổi tên collection theo timestamp để tránh bị cache ChromaDB cũ
    store = EmbeddingStore(collection_name=f"benchmark_{int(time.time())}", embedding_fn=embedder)
    store.add_documents(docs)
    
    # 3. Setup Agent với OpenAI Chat Completion
    client = OpenAI()
    def ask_llm(prompt: str) -> str:
        response = client.chat.completions.create(
            model="gpt-4o-mini", # Đệ có thể đổi thành gpt-3.5-turbo nếu thích
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return response.choices[0].message.content

    agent = KnowledgeBaseAgent(store=store, llm_fn=ask_llm)

    # 4. Tập Queries khốc liệt
    queries = [
        "Tôi có bao nhiêu ngày để gửi yêu cầu trả hàng hoàn tiền?",
        "Tiền hoàn về ví ShopeePay mất bao lâu?",
        "Đồng kiểm là gì và tôi được làm gì khi đồng kiểm?",
        "Nếu trả hàng theo hình thức tự sắp xếp, tôi có được hoàn phí vận chuyển không?",
        "Mã giảm giá có được hoàn lại khi tôi trả hàng toàn bộ đơn không?"
    ]

    print("\n" + "="*60)
    print("BẮT ĐẦU CHẠY BENCHMARK")
    print("="*60)

    for i, q in enumerate(queries, 1):
        print(f"\n[Q{i}]: {q}")
        
        # Manually search to get Top-1 details for the report
        results = store.search(q, top_k=3)
        top1 = results[0] if results else None
        
        if top1:
            print(f"  > Top-1 Score: {top1['score']:.4f}")
            print(f"  > Top-1 Child Chunk (Search): {top1['content']}")
            # Lấy tiêu đề của Parent để biết nó lôi luật nào ra
            print(f"  > Bối cảnh (Parent): {top1['metadata'].get('parent_title', 'Unknown')}")
            
        # Get final answer from LLM
        answer = agent.answer(q, top_k=3)
        print(f"  > Agent Trả lời: {answer.strip().replace(chr(10), ' ')}")
        print("-" * 40)

if __name__ == "__main__":
    run_benchmark()