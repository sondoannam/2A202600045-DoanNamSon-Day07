import os
from src.chunking import ChunkingStrategyComparator
from src.chunking import ParentChildChunker 

def run_comparison():
    # file_path = "data/shopee_dong_kiem.md"
    # file_path = "data/shopee_chinh_sach_tra_hang_hoan_tien.md"
    file_path = "data/shopee_huy_don_hoan_voucher.md"
    
    if not os.path.exists(file_path):
        print(f"Không tìm thấy file: {file_path}")
        return
        
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
        
    print("="*50)
    print("1. BASELINE STRATEGIES")
    print("="*50)
    
    comparator = ChunkingStrategyComparator()
    # Chạy so sánh 3 baseline với chunk_size tiêu chuẩn
    baseline_results = comparator.compare(text, chunk_size=300)
    
    for name, stats in baseline_results.items():
        print(f"❖ Strategy: {name}")
        print(f"  - Chunk Count: {stats['count']}")
        print(f"  - Avg Length: {stats['avg_length']:.1f} ký tự")
        # print(f"  - Sample: {stats['chunks'][0][:80]}...\n")
        sample_chunk = stats['chunks'][20] if len(stats['chunks']) > 20 else stats['chunks'][0]
        print(f"  - Sample: {sample_chunk}\n")

    print("="*50)
    print("2. CUSTOM STRATEGY: PARENT-CHILD")
    print("="*50)
    
    custom_chunker = ParentChildChunker(max_sentences_per_child=2)
    custom_chunks = custom_chunker.chunk_with_metadata(text)
    
    custom_count = len(custom_chunks)
    # Tính trung bình độ dài của phần "content" (Child chunk)
    custom_avg_length = sum(len(c["content"]) for c in custom_chunks) / custom_count if custom_count > 0 else 0
    
    print(f"❖ Strategy: parent_child_shopee")
    print(f"  - Chunk Count: {custom_count}")
    print(f"  - Avg Length: {custom_avg_length:.1f} ký tự")
    print(f"  - Sample Content (Dùng để Search): {custom_chunks[1]['content']}")
    print(f"  - Sample Metadata (Dùng để đưa cho LLM): {str(custom_chunks[1]['metadata']['parent_content'])[:80]}...\n")

if __name__ == "__main__":
    run_comparison()