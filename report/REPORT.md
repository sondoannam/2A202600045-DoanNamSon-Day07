# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Đoàn Nam Sơn
**Nhóm:** C401-F2
**Ngày:** 10/4/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
- Khi hai đoạn văn bản có độ tương đồng cosin cao, điều đó có nghĩa là ý nghĩa ngữ nghĩa (semantic meaning) của chúng rất giống nhau, cùng hướng về một chủ đề trong không gian vector, dù độ dài hay từ vựng có thể không giống nhau y hệt.

**Ví dụ HIGH similarity:**
- Sentence A: "Tôi muốn tìm hiểu về AI và machine learning"
- Sentence B: "Tôi quan tâm đến trí tuệ nhân tạo và học máy"
- Tại sao tương đồng: Cả hai câu đều thể hiện sự quan tâm đến cùng một lĩnh vực (AI và machine learning), mặc dù sử dụng từ ngữ khác nhau.

**Ví dụ LOW similarity:**
- Sentence A: "Tôi muốn tìm hiểu về AI và machine learning"
- Sentence B: "Thời tiết hôm nay đẹp quá"
- Tại sao khác: Hai câu này không có bất kỳ sự liên quan nào về mặt ngữ nghĩa.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
- Cosine similarity đo góc giữa hai vector, tập trung vào hướng (ý nghĩa) thay vì độ lớn (độ dài văn bản). Điều này giúp so sánh ngữ nghĩa hiệu quả hơn, đặc biệt khi văn bản có độ dài khác nhau.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
- *Trình bày phép tính:* 
    > $num\_chunks = \lceil (10000 - 50) / (500 - 50) \rceil = \lceil 9950 / 450 \rceil$
- *Đáp án:* 23 chunks

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
- *Phép tính:* 
    > $num\_chunks = \lceil (10000 - 100) / (500 - 100) \rceil = \lceil 9900 / 400 \rceil = 25$ chunks
- *Đáp án:* 25 chunks
- *Tại sao muốn overlap nhiều hơn:* 
    > Khi overlap tăng lên, số lượng chunks tăng lên. Điều này là do mỗi chunk có nhiều nội dung được chia sẻ với chunk trước đó và chunk sau đó, giúp mô hình có nhiều ngữ cảnh hơn để hiểu ý nghĩa của từng chunk.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Shopee — Chính sách Trả hàng / Hoàn tiền (FAQ)

**Tại sao nhóm chọn domain này?**
> Shopee là sàn TMĐT phổ biến tại Việt Nam, chính sách trả hàng/hoàn tiền là nội dung người dùng thường xuyên cần tra cứu. Domain này có cấu trúc FAQ rõ ràng, nhiều điều kiện cụ thể dễ đánh giá độ chính xác của retrieval. Ngoài ra tài liệu tiếng Việt giúp nhóm thực hành RAG với ngôn ngữ thực tế.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | shopee_chinh_sach_tra_hang_hoan_tien.md | help.shopee.vn/portal/4/article/77251 | 27,287 | category: "policy", topic: "return_refund", lang: "vi" |
| 2 | shopee_dong_kiem.md | help.shopee.vn/portal/4/article/124982 | 9,948 | category: "faq", topic: "dong_kiem", lang: "vi" |
| 3 | shopee_huy_don_hoan_voucher.md | help.shopee.vn/portal/4/article/79296 | 7,641 | category: "faq", topic: "voucher_refund", lang: "vi" |
| 4 | shopee_phuong_thuc_tra_hang.md | help.shopee.vn/portal/4/article/189477 | 9,467 | category: "guide", topic: "return_method", lang: "vi" |
| 5 | shopee_quy_dinh_chung_tra_hang.md | help.shopee.vn | 9,014 | category: "policy", topic: "return_rules", lang: "vi" |
| 6 | shopee_thoi_gian_hoan_tien.md | help.shopee.vn/portal/4/article/189473 | 7,346 | category: "faq", topic: "refund_timeline", lang: "vi" |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| category | string | "policy", "faq", "guide" | Phân loại tài liệu — filter khi muốn chỉ tìm FAQ hoặc policy |
| topic | string | "return_refund", "dong_kiem", "refund_timeline" | Narrow scope retrieval theo chủ đề cụ thể |
| lang | string | "vi" | Hữu ích nếu sau này mở rộng sang tài liệu tiếng Anh |


---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| data/shopee_chinh_sach_tra_hang_hoan_tien.md | FixedSizeChunker (`fixed_size`) | 76 | 296.7 | |
| data/shopee_chinh_sach_tra_hang_hoan_tien.md | SentenceChunker (`by_sentences`) | 56 | 372.9 | |
| data/shopee_chinh_sach_tra_hang_hoan_tien.md | RecursiveChunker (`recursive`) | 140 | 194.2 | |
<!-- | data/shopee_chinh_sach_tra_hang_hoan_tien.md | ParentChildChunker (`parent_child`) | 92 | 226.7 | | -->
| data/shopee_dong_kiem.md | FixedSizeChunker (`fixed_size`) | 29 | 298.7 | |
| data/shopee_dong_kiem.md | SentenceChunker (`by_sentences`) | 10 | 807.2 | |
| data/shopee_dong_kiem.md | RecursiveChunker (`recursive`) | 63 | 164.1 | |
<!-- | data/shopee_dong_kiem.md | ParentChildChunker (`parent_child`) | 14 | 576.3 | | -->
| data/shopee_huy_don_hoan_voucher.md | FixedSizeChunker (`fixed_size`) | 24 | 294.4 | |
| data/shopee_huy_don_hoan_voucher.md | SentenceChunker (`by_sentences`) | 4 | 1649.2 | |
| data/shopee_huy_don_hoan_voucher.md | RecursiveChunker (`recursive`) | 49 | 198.6 | |
<!-- | data/shopee_huy_don_hoan_voucher.md | ParentChildChunker (`parent_child`) | 9 | 410.2 | | -->

### Strategy Của Tôi

**Loại:** ParentChildChunker (`parent_child`)

**Mô tả cách hoạt động:**
- Nhận diện các tiêu đề chính (Parent) thông qua Regex. Sau đó băm nội dung bên trong thành các câu ngắn hoặc tách bảng theo từng dòng (Child). Khi lưu vào vector store, chỉ dùng Child để nhúng (embed), nhưng gắn toàn bộ Parent text vào metadata.

**Tại sao tôi chọn strategy này cho domain nhóm?**
- Tài liệu FAQ chính sách có cấu trúc rất rõ ràng theo từng Điều khoản, nhưng chứa nhiều điều kiện/lưu ý chéo nhau. Nếu băm nhỏ quá sẽ mất điều kiện lưu ý, nếu băm lớn quá sẽ khó truy xuất đúng. Parent-Child giúp tìm kiếm cực nhạy (nhờ Child) nhưng giữ bối cảnh hoàn hảo cho LLM (nhờ Parent).

**Code snippet (nếu custom):**
```python
class ParentChildChunker:
    """
    Custom Parent-Child chunking strategy specifically designed for Shopee Policies.
    It identifies major policy sections (Parents) and splits them into smaller, 
    embeddable sentences or table rows (Children) while preserving the parent context.
    """
    def __init__(self, max_sentences_per_child: int = 2):
        self.max_sentences_per_child = max_sentences_per_child

    def chunk_with_metadata(self, text: str) -> List[Dict[str, Any]]:
        """
        Splits text into children and binds them with their parent metadata.
        Returns a list of dictionaries containing 'content' (child) and 'metadata' (parent info).
        """
        # Regex to detect section headers like "1.1. Đối Tượng Áp Dụng" or "***4.1. Đối tượng...***"
        # Matches start of line, optional asterisks, numbers separated by dots.
        pattern = re.compile(r'^(?:\*\*\*)?\d+\.\d+.*$', re.MULTILINE)
        
        matches = list(pattern.finditer(text))
        chunks = []
        
        # Handle cases where no clear sections are found
        if not matches:
            return self._process_parent("General Policy", text)

        # Process any introductory text before the first section
        if matches[0].start() > 0:
            intro_text = text[:matches[0].start()].strip()
            if intro_text:
                chunks.extend(self._process_parent("Introduction", intro_text))

        # Process each detected parent section
        for i in range(len(matches)):
            start_idx = matches[i].start()
            # The section ends where the next one begins, or at the end of the text
            end_idx = matches[i+1].start() if i + 1 < len(matches) else len(text)
            
            parent_block = text[start_idx:end_idx].strip()
            # Clean up the title (remove markdown bold/italic asterisks)
            parent_title = matches[i].group().strip('* ')
            
            chunks.extend(self._process_parent(parent_title, parent_block))
            
        return chunks

    def _process_parent(self, parent_title: str, parent_text: str) -> List[Dict[str, Any]]:
        """
        Internal helper to split a Parent block into Child chunks based on its structure (Table vs Text).
        """
        children = []
        
        # 1. SPECIAL CASE: Markdown Tables
        if "|" in parent_text and "\n|---" in parent_text.replace(" ", ""):
            lines = parent_text.split('\n')
            headers = []
            for line in lines:
                if line.strip().startswith('|'):
                    if '---' in line:
                        continue # Skip the separator line
                    
                    # Extract cell values, ignoring empty strings from split
                    cells = [c.strip() for c in line.split('|')[1:-1]]
                    
                    if not headers:
                        headers = cells # The first valid row is the header
                    else:
                        # Combine column headers with cell values to create a meaningful child sentence
                        child_text = ". ".join(
                            [f"{headers[i]}: {cells[i]}" for i in range(min(len(headers), len(cells))) if cells[i]]
                        )
                        if child_text:
                            children.append(child_text)
                            
        # 2. NORMAL CASE: Regular Text
        else:
            # Fallback to standard sentence splitting using regex
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\.\n', parent_text) if s.strip()]
            
            for i in range(0, len(sentences), self.max_sentences_per_child):
                child_text = " ".join(sentences[i:i + self.max_sentences_per_child])
                children.append(child_text)
                
        # 3. METADATA BINDING: Wrap each child with its parent's DNA
        results = []
        for child in children:
            if child:  # Ensure no empty chunks are added
                results.append({
                    "content": child,  # This will be embedded
                    "metadata": {
                        "parent_title": parent_title,
                        "parent_content": parent_text,  # This will be given to the LLM
                        "strategy": "parent_child"
                    }
                })
        return results
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| data/shopee_chinh_sach_tra_hang_hoan_tien.md | best baseline | 140 | 194.2 | |
| data/shopee_chinh_sach_tra_hang_hoan_tien.md | **của tôi** | 92 | 296.2 | |
| data/shopee_dong_kiem.md | best baseline | 63 | 164.1 | |
| data/shopee_dong_kiem.md | **của tôi** | 14 | 576.3 | |
| data/shopee_huy_don_hoan_voucher.md | best baseline | 49 | 198.6 | |
| data/shopee_huy_don_hoan_voucher.md | **của tôi** | 9 | 410.2 | |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Giang | Custome Recursive Strategy | 8 | Chunking dựa theo cấu trúc của tài liệu, đảm bảo tính toàn vẹn của thông tin đoạn văn | với những đoạn dài chunk có thể vượt quá lượng ký tự cho phép của mô hình embedding |
| Nhữ Gia Bách | SemanticChunker | 8/10 | Chunk đúng chủ đề, score distribution rõ | Thiếu thông tin số liệu cụ thể khi chunk tách rời context |
| Trần Quang Quí | DocumentStructureChunker| 9/10 (5/5 relevant, avg score 0.628) | Chunk bám sát cấu trúc Q&A, context coherent, không bị cắt giữa điều khoản | Multi-aspect query (Q3) score thấp 0.59 vì định nghĩa và hướng dẫn nằm ở 2 chunk khác nhau |
| Vũ Đức Duy | Agentic Chunker | 9/10 (Avg: 0.669) | Gom ngữ nghĩa cực sâu, chia văn bản rành mạch (cắt gọn giảm 4 lần lượng chunks thừa). Score cao nhất. | Phải gọi API tốn kém kinh phí, index siêu chậm, phụ thuộc vào chất lượng parser. |
| Tôi (Đoàn Nam Sơn) | Parent-Child Chunking | 9/10 (5/5 relevant, avg score 0.66) | Chunk hoạt động rất tốt, cắt đúng theo pattern Q&A, không bị cắt giữa điều khoản, rất phù hợp đối với các tài liệu có cấu trúc rõ ràng. | Với các tài liệu không có cấu trúc rõ ràng thì có thể không hiệu quả. |


**Strategy nào tốt nhất cho domain này? Tại sao?**
- Strategy tốt nhất cho domain này là Strategy của tôi (Đoàn Nam Sơn) vì thể hiện rõ sự hiệu quả đối với các tài liệu có cấu trúc rõ ràng như tài liệu này. Tuy nhiên, với các tài liệu không có cấu trúc rõ ràng thì có thể không hiệu quả.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
- Thuật toán sử dụng Regex `(?<=[.!?])\s+|\.\n` kết hợp Lookbehind để phát hiện chính xác ranh giới câu (dấu chấm, hỏi, than đi kèm khoảng trắng hoặc xuống dòng) mà không làm mất đi dấu câu đó. Edge case được xử lý là loại bỏ các khoảng trắng thừa (`strip()`) và chặn các chuỗi rỗng trước khi gom nhóm các câu lại theo kích thước `max_sentences_per_chunk`.

**`RecursiveChunker.chunk` / `_split`** — approach:
- Thuật toán hoạt động theo cơ chế đệ quy (recursion) duyệt qua danh sách các dấu phân cách (separators) theo mức độ ưu tiên. Base case (điều kiện dừng) là khi độ dài đoạn văn bản hiện tại đã nhỏ hơn `chunk_size` hoặc không còn separator nào để thử. Nếu một đoạn được cắt ra vẫn lớn hơn `chunk_size`, hàm sẽ tự động gọi lại chính nó (`_split`) với cấp độ separator nhỏ hơn tiếp theo để tiếp tục phân rã.

### EmbeddingStore

**`add_documents` + `search`** — approach:
- Hệ thống được thiết kế ưu tiên sử dụng `ChromaDB` để tối ưu hiệu năng lưu trữ vector, với cơ chế fallback về `in-memory list` nếu môi trường không hỗ trợ. `add_documents` sẽ trích xuất ID, tính toán embedding và ép kiểu metadata (xử lý None thành dict rỗng) trước khi nạp vào store. Hàm search nhúng (embed) câu truy vấn, tìm kiếm thông qua `collection.query()`, sau đó đảo ngược khoảng cách (distance) thành độ tương đồng `(score = 1.0 - distance)` để chuẩn hóa output đầu ra.

**`search_with_filter` + `delete_document`** — approach:
- Việc lọc (filter) luôn được thực hiện trước khi tìm kiếm vector thông qua tham số `where` của ChromaDB (hoặc lọc list trong bộ nhớ trước khi tính cosine similarity), giúp giảm không gian tìm kiếm và tăng độ chính xác. `delete_document` thực hiện xóa dữ liệu đa luồng: xóa theo ID gốc của document và xóa theo trường `doc_id` bên trong metadata để dọn dẹp triệt để các chunk con bị phân tách.

### KnowledgeBaseAgent

**`answer`** — approach:
- Luồng RAG được thực thi qua 3 bước: Retrieve (gọi `store.search` lấy `top_k` chunks) -> Xây dựng Prompt -> Sinh câu trả lời (Call LLM). Context được inject vào prompt bằng cách nối (join) phần `content` của các chunk trả về thành một khối văn bản thống nhất, đặt trước câu hỏi của người dùng để ép LLM phải căn cứ (grounding) vào tài liệu nội bộ thay vì dùng kiến thức nền.

### Test Results

```text
❯ pytest tests/ -v
======================================================== test session starts ========================================================
platform darwin -- Python 3.11.15, pytest-9.0.2, pluggy-1.6.0 -- /Users/sondoannam/miniforge3/envs/vinuni_ai/bin/python3.11
cachedir: .pytest_cache
rootdir: /Users/sondoannam/vinuni/Day07-C401-F2
plugins: langsmith-0.7.26, anyio-4.13.0
collected 42 items                                                                                                                  

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED                                         [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED                                                  [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED                                           [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED                                            [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED                                                 [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED                                 [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED                                       [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED                                        [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED                                      [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED                                                        [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED                                        [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED                                                   [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED                                               [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED                                                         [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED                                [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED                                    [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED                              [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED                                    [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED                                                        [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED                                          [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED                                            [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED                                                  [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED                                       [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED                                         [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED                             [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED                                          [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED                                                   [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED                                                  [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED                                             [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED                                         [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED                                    [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED                                        [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED                                              [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED                                        [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED                     [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED                                   [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED                                  [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED                      [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED                                 [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED                          [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED                [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED                    [100%]

======================================================== 42 passed in 0.83s =========================================================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Tôi có bao nhiêu ngày để gửi yêu cầu trả hàng hoàn tiền? | Người Mua có thể gửi yêu cầu trả hàng/hoàn tiền trong vòng 15 (mười lăm) ngày kể từ lúc đơn hàng được cập nhật giao hàng... | High | 0.70 | Yes |
| 2 | Tiền hoàn về ví ShopeePay mất bao lâu? | : Thanh toán khi nhận hàng/Chuyển khoản ngân hàng. : Ví ShopeePay - áp dụng từ ngày 13.11.2025. : 24 giờ... | High | 0.66 | Yes |
| 3 | Đồng kiểm là gì và tôi được làm gì khi đồng kiểm? | Theo hình thức “Tự sắp xếp”: Người Mua cần thanh toán trước chi phí vận chuyển cho việc trả hàng. | Low | 0.12 | Yes |
| 4 | Mã giảm giá có được hoàn lại khi tôi trả hàng toàn bộ đơn không? | Tiền hoàn sẽ được chuyển vào Ví ShopeePay, SPayLater, Thẻ nội địa Napas, Tài Khoản Ngân Hàng... | Low | 0.21 | Yes |
| 5 | Đồng kiểm là gì và tôi được làm gì khi đồng kiểm? | Khi đồng kiểm với Bưu tá, bạn chỉ được kiểm tra ngoại quan... KHÔNG được mở tem... KHÔNG được sử dụng thử... | High | 0.55 | Yes |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
- Kết quả bất ngờ nhất là Cặp số 5 (Hỏi về Đồng kiểm). Dù Chunk B chứa thông tin giải quyết hoàn hảo câu hỏi A, nhưng điểm tương đồng (0.55) lại thấp hơn hẳn so với Cặp số 1 (0.70). Điều này cho thấy Embeddings không chỉ so khớp từ khóa (lexical) mà còn cực kỳ nhạy cảm với "ý định" (intent) của câu: Câu A mang ý định "hỏi định nghĩa/khái niệm", trong khi Câu B mang ý định "hướng dẫn quy trình", sự lệch pha về ý định ngữ nghĩa này khiến vector của chúng không hoàn toàn nằm sát nhau trong không gian.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Tôi có bao nhiêu ngày để gửi yêu cầu trả hàng hoàn tiền? | 15 ngày kể từ lúc đơn hàng được cập nhật trạng thái Giao hàng thành công. |
| 2 | Tiền hoàn về ví ShopeePay mất bao lâu? | 24 giờ (với điều kiện Ví ShopeePay vẫn hoạt động bình thường). |
| 3 | Đồng kiểm là gì và tôi được làm gì khi đồng kiểm? | Kiểm tra ngoại quan và số lượng sản phẩm khi nhận hàng. Không được mở tem, dùng thử. |
| 4 | Nếu trả hàng theo hình thức tự sắp xếp, tôi có được hoàn phí vận chuyển không? | Có, Shopee hoàn lại trong 3-5 ngày làm việc (hoặc Shopee Xu với đơn ngoài Mall). |
| 5 | Mã giảm giá có được hoàn lại khi tôi trả hàng toàn bộ đơn không? | Có, mã giảm giá được hoàn nếu khiếu nại toàn bộ sản phẩm và được chấp nhận hoàn tiền. |


### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Tôi có bao nhiêu ngày để gửi yêu cầu trả hàng hoàn tiền? | Người Mua có thể gửi yêu cầu trả hàng/hoàn tiền trong vòng 15 ngày... | 0.7025 | YES | 3.2. Người Mua có thể gửi yêu cầu trả hàng/hoàn tiền trong vòng 15 (mười lăm) ngày kể từ lúc đơn hàng được cập nhật giao hàng thành công. |
| 2 | Tiền hoàn về ví ShopeePay mất bao lâu? | ... Ví ShopeePay - áp dụng từ ngày 13.11.2025. : 24 giờ... | 0.6633 | YES | Tiền hoàn về ví ShopeePay sẽ được nhận trong 24 giờ (nếu ví bình thường). |
| 3 | Đồng kiểm là gì và tôi được làm gì khi đồng kiểm? | Khi đồng kiểm... chỉ được kiểm tra ngoại quan... KHÔNG được mở tem... | 0.5573 | YES | Cho phép kiểm tra ngoại quan và số lượng. Không được bóc tem, dùng thử hay làm hỏng sản phẩm. |
| 4 | Nếu trả hàng theo hình thức tự sắp xếp, tôi có được hoàn phí vận chuyển không? | Theo hình thức “Tự sắp xếp”: Người Mua cần thanh toán trước chi phí... | 0.6983 | YES | Cần thanh toán trước, Shopee sẽ hoàn lại một phần bằng Shopee Xu nếu đáp ứng điều kiện. |
| 5 | Mã giảm giá có được hoàn lại khi tôi trả hàng toàn bộ đơn không? | Mã giảm giá (Voucher) có thể Được/Không được hoàn lại tùy theo quy định... | 0.6794 | YES | Có thể được hoặc không được hoàn lại tùy theo quy định dành cho đơn Hủy/Trả hàng. |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
- Qua chiến lược Semantic của Bách và DocumentStructure của Quí, tôi nhận ra "điểm yếu" của các phương pháp băm dữ liệu truyền thống. Dù băm theo ngữ nghĩa (Semantic) hay theo cấu trúc tĩnh, hệ thống vẫn rất dễ đánh mất các con số/số liệu cụ thể khi tách rời khỏi bối cảnh gốc, đồng thời gặp khó khăn lớn với các truy vấn đa khía cạnh (multi-aspect queries như Q3) khi thông tin nằm rải rác. Điều này càng chứng minh việc giữ lại "Parent context" là vô cùng thiết yếu để duy trì tính vẹn toàn của thông tin.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
- Tôi nhận ra rằng "Data Quality > Model Choice" (Chất lượng dữ liệu quan trọng hơn việc chọn mô hình). Nhiều nhóm gặp thất bại không phải do thuật toán chunking kém, mà do dữ liệu đầu vào chứa quá nhiều rác (formatting lỗi, thiếu nhất quán). Ngoài ra, việc thiết kế Metadata Schema thông minh để phân loại (filter) ngay từ đầu giúp thu hẹp không gian tìm kiếm và hạn chế LLM bị "ảo giác" (hallucinate) hiệu quả hơn rất nhiều so với việc chỉ dùng Vector Search thuần túy.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
- Tôi sẽ đầu tư mạnh hơn vào khâu Tiền xử lý (Data Preprocessing) và Làm giàu dữ liệu (Data Enrichment). Thay vì chỉ băm tài liệu có sẵn, tôi sẽ dùng một LLM nhỏ chạy offline để tóm tắt hoặc sinh ra các "Câu hỏi giả định" (Hypothetical Questions) và nhét chúng vào Metadata của từng Parent Chunk. Khi người dùng đặt câu hỏi, việc tính toán độ tương đồng giữa Câu hỏi - Câu hỏi sẽ mang lại độ nhạy (Precision) cao hơn rất nhiều so với Câu hỏi - Đoạn văn thông thường.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 15 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 5 / 5 |
| **Tổng** | | **100 / 100** |