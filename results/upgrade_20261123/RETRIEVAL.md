# Truy xuất: tái lập Dev100 và kiểm tra dấu ngoặc kép Unicode

## Kết quả chính và phạm vi kết luận

Trên Dev100 chuẩn, Hybrid dùng cosine chính xác đạt **55/100 tại Recall@5**, giữ nguyên số lượt đúng và tập câu đúng ở các mốc 1/3/5/10 so với kết quả exact đã công bố trên Windows. Dense được đo lại trong cùng phiên Linux đạt **50/100**. Cặp so sánh cùng phiên là **50% → 55%, tăng 5 điểm phần trăm**; đây là chênh lệch Hybrid so với Dense, không phải mức tăng mới của exact sau lần sửa này. [Kết quả Linux](retrieval/dev100.json) · [Kết quả Windows đã lưu](../colab/retrieval/dev100.json).

Trên toàn bộ 50 câu không dấu đã dùng trong phát triển, thêm dấu ngoặc kép cong `“…”` từng làm truy xuất giảm mạnh. Sau sửa, Recall@5 của biến thể này tăng từ **7/50 lên 22/50**, tức **14% → 44%, tăng 30 điểm phần trăm**, với **15 câu thắng và 0 câu thua**. Cùng 50 câu ở dạng không dấu ban đầu vẫn đạt **22/50**, không thay đổi. Đây là cải thiện độ bền trước một dạng dấu câu trên những ví dụ đã biết; không phải mức tăng trên 100 câu mới, một holdout độc lập hoặc Dev100 chuẩn. [Trước sửa](robustness/before.json) · [Sau sửa](robustness/after.json).

Các kết quả trong tài liệu này chỉ đánh giá tìm tài liệu. Chúng không đo độ đúng y khoa, không thay thế đánh giá generation và không xác nhận toàn bộ bản nộp đã hoàn tất. Tất cả lượt được trình bày ở đây chạy trên **Linux CPU cục bộ, chưa phải hosted Google Colab**.

## 1. Cấu hình và nguồn gốc phép đo

Giữ nguyên ViHealthQA revision `d90a62deedfc822a15de3f35535bb41474e4d3f4`, E5-small revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`, vector chuẩn hóa, chunk 200 từ và overlap 40. Kho có 9.895 tài liệu, 11.227 đoạn; không thêm câu hỏi hay nhãn gold vào nội dung lập chỉ mục. Chữ ký corpus của cả hai nhóm đánh giá là:

```text
35902972d43e0841495d6f920901c99146d08e7e745cb5670c76d3d9f367ea87
```

| Đường truy xuất | Ngân sách ứng viên |
| --- | --- |
| `dense` | HNSW 100 đoạn |
| `hybrid_v2` | HNSW 80 đoạn, giữ 80 |
| `locked` | HNSW 200 đoạn, giữ 80 |
| `selected` | Cosine chính xác trên vector đã lưu, lấy 200 đoạn, giữ 80 |

Mỗi đường Hybrid kết hợp BM25 80 tài liệu bằng RRF `k=40`, trọng số 0,5 cho mỗi nhánh. Mốc K được tính theo tài liệu sau khử trùng. Phép đo gọi truy xuất với `k=10` rồi tính 1/3/5/10 từ cùng thứ hạng; không tăng nhóm ứng viên riêng cho từng mốc. Dev100 chạy các phương pháp luân phiên theo câu, sau ba câu warm-up cho mỗi phương pháp. [Giao thức](PROTOCOL.md) · [Metadata của Dev100](retrieval/dev100.json).

Môi trường được ghi nhận: Linux x86_64, Python 3.11.14, PyTorch `2.7.1+cpu`, sentence-transformers 3.4.1, Chroma 1.0.20, NumPy 1.26.4; E5 dùng CPU, Torch 4 thread và OpenBLAS 1 thread. Coverage bằng 100%, lỗi ánh xạ bằng 0. Query của Dev100 và tập xác nhận đều không bị cắt ở giới hạn 512 token trong phép đo tokenizer có tính tiền tố và special token. [Metadata Dev100](retrieval/dev100.json) · [Metadata xác nhận](retrieval/confirmation100.json).

## 2. Dev100: đo mới trong cùng phiên Linux

Mỗi ô Recall dưới đây ghi **số câu đúng / 100**. Toàn bộ 100 ID được giữ, bao gồm các câu không tìm được gold.

| Phương pháp | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Coverage | Lỗi |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense | 36/100 | 46/100 | 50/100 | 54/100 | 100% | 0 |
| Hybrid v2 | 37/100 | 51/100 | 55/100 | 59/100 | 100% | 0 |
| Hybrid HNSW đã khóa | 38/100 | 52/100 | 55/100 | 60/100 | 100% | 0 |
| Hybrid exact đang chọn | 38/100 | 52/100 | 55/100 | 60/100 | 100% | 0 |

Nguồn: [bảng tổng hợp](retrieval/dev100.json) và [kết quả từng câu/phương pháp](retrieval/dev100.jsonl).

### Đối chiếu theo từng câu: Dense → Hybrid exact

Một câu thắng khi Dense không tìm được gold trong top K nhưng exact tìm được; một câu thua là trường hợp ngược lại. Đối chiếu thực hiện trên toàn bộ 100 ID, xác minh cùng `source_id` và cùng tập `gold_doc_ids`.

| Mốc | Dense đúng | Exact đúng | Thắng | Thua | Cả hai đúng | Cả hai trượt |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| @1 | 36 | 38 | 8 | 6 | 30 | 56 |
| @3 | 46 | 52 | 9 | 3 | 43 | 45 |
| @5 | 50 | 55 | 7 | 2 | 48 | 43 |
| @10 | 54 | 60 | 8 | 2 | 52 | 38 |

Tại @5, các ID thắng là `d000`, `d005`, `d027`, `d028`, `d030`, `d081`, `d098`; các ID thua là `d007`, `d034`. Không suy ra mỗi câu đều tốt hơn từ mức tăng ròng 5 câu. [Các dòng gốc để đối chiếu](retrieval/dev100.jsonl).

Trong cùng phiên, `locked` và `selected` có **0 thắng / 0 thua ở cả bốn mốc**. So với `hybrid_v2`, `selected` có thắng/thua lần lượt **1/0, 1/0, 1/1, 1/0** tại @1/3/5/10. Riêng @5, exact thêm `d000` nhưng mất `d071`, nên hai phương pháp có cùng tổng 55 dù tập câu đúng không hoàn toàn giống nhau.

Đối chiếu exact trên [Windows đã lưu](../colab/retrieval/dev100.jsonl) với exact [Linux mới](retrieval/dev100.jsonl) cho **0 thắng / 0 thua trên cả 100 câu ở mọi mốc 1/3/5/10**. Đây là kết quả tái lập giữa hai môi trường, không phải phép cô lập tác động thuật toán. Không dùng Dense 51% hoặc Hybrid v2 56% của lượt Windows cũ để gán chênh lệch cho bản sửa Linux; cặp so sánh trong bảng này dùng những số đo Linux vừa thực hiện.

### Độ trễ retrieval

| Phương pháp | Trung bình (ms) | Trung vị (ms) | p95 (ms) |
| --- | ---: | ---: | ---: |
| Dense | 23,280 | 20,793 | 33,049 |
| Hybrid v2 | 153,030 | 107,064 | 344,114 |
| Hybrid HNSW đã khóa | 148,011 | 107,608 | 324,134 |
| Hybrid exact đang chọn | 142,503 | 99,924 | 340,171 |

Các số này bao gồm embedding truy vấn và xử lý truy xuất, không bao gồm sinh câu trả lời. Chi phí mở/xác minh index được ghi riêng là 27,729 giây; RSS cao nhất của tiến trình đo khoảng 1.588,3 MiB, không phải phép đo bộ nhớ tách riêng từng phương pháp. Exact có trung vị thấp hơn `locked` trong lượt này nhưng p95 cao hơn, nên không kết luận exact luôn nhanh hơn. [Metadata và thống kê thời gian](retrieval/dev100.json).

## 3. Tái lập tập `confirmation100` đã công khai

Đây là đúng 100 ID trong [manifest bộ câu hỏi](../colab/retrieval/sets.json). Tập này đã được công khai trước đợt sửa hiện tại; kết quả dưới đây là tái lập trên tập đã biết, không phải xác nhận độc lập mới.

| Phương pháp | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Coverage | Lỗi |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hybrid HNSW đã khóa | 49/100 | 62/100 | 68/100 | 75/100 | 100% | 0 |
| Hybrid exact đang chọn | 48/100 | 62/100 | 68/100 | 75/100 | 100% | 0 |

Exact so với `locked` có **0 thắng / 1 thua ở @1** và **0 thắng / 0 thua ở @3/5/10**, xét đủ 100 câu. Exact tái lập đúng các hit/miss của [kết quả Windows trước đó](../colab/retrieval/confirmation100.jsonl) tại cả bốn mốc: **0 thắng / 0 thua**. Không ghép kết quả của tập này với những tập xác nhận khác. [Bảng Linux](retrieval/confirmation100.json) · [Từng câu Linux](retrieval/confirmation100.jsonl).

## 4. Sửa lỗi định tuyến câu không dấu có dấu câu Unicode

Điều kiện cũ coi mọi ký tự có mã lớn hơn 127 là dấu tiếng Việt. Vì vậy, chỉ thêm `“` và `”` cũng làm một câu không dấu đi vào nhánh token hóa có dấu. Bản sửa chuẩn hóa NFC và nhận diện đúng tập chữ tiếng Việt có dấu, gồm Đ/đ; dấu câu và emoji không tự quyết định nhánh ngôn ngữ. Đồng thời, việc chọn đoạn đại diện cho tài liệu được tách khỏi điểm và thứ hạng BM25/RRF. [Mã truy xuất](../../src/retrieval.py) · [Phạm vi thay đổi](PROTOCOL.md).

Phép đo sử dụng cùng evaluator ở hai checkout, toàn bộ 50 ID NoDiac50 đã khóa, với hai biến thể duy nhất:

1. `original`: câu không dấu được loader tái tạo.
2. `curly_quotes`: chính câu đó, chỉ thêm U+201C và U+201D ở hai đầu.

Như vậy mỗi tệp có 100 dòng kết quả nhưng chỉ có **50 câu gốc**, không phải 100 câu độc lập. Mỗi biến thể giữ mẫu số 50; lỗi được tính là miss. Không gọi LLM, không đưa gold vào truy vấn và không thay tham số sau khi xem kết quả phép đo này. [Giao thức đã xác định trước phép đo Unicode](PROTOCOL.md) · [Evaluator](../../benchmarks/evaluate_unicode_robustness.py).

### Số lượt đúng ở từng mốc

| Biến thể | Phiên bản | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Lỗi |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Không dấu gốc | Trước sửa | 8/50 | 20/50 | 22/50 | 24/50 | 0 |
| Không dấu gốc | Sau sửa | 8/50 | 20/50 | 22/50 | 24/50 | 0 |
| Không dấu + `“…”` | Trước sửa | 3/50 | 5/50 | 7/50 | 7/50 | 0 |
| Không dấu + `“…”` | Sau sửa | 7/50 | 18/50 | 22/50 | 24/50 | 0 |

Nguồn: [JSON trước sửa](robustness/before.json) và [JSON sau sửa](robustness/after.json), bao gồm các dòng ID/gold/retrieved tương ứng.

### Đối chiếu trước/sau trên đủ 50 câu có ngoặc kép

| Mốc | Đúng trước | Đúng sau | Thắng | Thua | Cả hai đúng | Cả hai trượt |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| @1 | 3 | 7 | 4 | 0 | 3 | 43 |
| @3 | 5 | 18 | 13 | 0 | 5 | 32 |
| @5 | 7 | 22 | 15 | 0 | 7 | 28 |
| @10 | 7 | 24 | 17 | 0 | 7 | 26 |

Các ID thắng tại @5: `t002`, `t004`, `t008`, `t011`, `t013`, `t018`, `t020`, `t022`, `t031`, `t032`, `t033`, `t034`, `t042`, `t044`, `t049`. Không có ID thua tại @5. Dạng không dấu gốc có **0 thắng / 0 thua ở mọi mốc** và giữ nguyên toàn bộ danh sách top 10 của cả 50 câu trước/sau.

Sau sửa, biến thể có ngoặc kép đạt cùng tổng 22/50 tại @5 như dạng gốc, nhưng không hoàn toàn bất biến với dấu câu: so với dạng gốc sau sửa, nó thêm `t002` và mất `t012` ở @5; @1 đạt 7 thay vì 8, @3 đạt 18 thay vì 20. Sửa nhánh token hóa không bảo đảm embedding hoặc thứ hạng sẽ giống nhau khi ký tự đầu vào thay đổi.

**Mức tăng 30 điểm phần trăm chỉ áp dụng cho biến thể ngoặc kép trên 50 ví dụ đã từng dùng phát triển.** Nó không chứng minh Recall@5 chuẩn tăng từ 55%, không chứng minh khả năng tổng quát hóa trên câu hỏi mới và không phải mức tăng độ chính xác y khoa.

## 5. Provenance, trạng thái chạy và tái lập

Trong cặp đo Unicode, `source_commit` ghi mốc HEAD tương ứng là `b9a586aecf9837fdf991dddd55f07d26b43d3b64` và `af282252ed56190a15d12c17eb258f3bc875f844`. Mốc HEAD không tự bao gồm thay đổi chưa commit: hash file trong artifact mới nhận diện bản thực thi. Trong danh sách các file phụ thuộc được ghi cho phép đo này, chỉ `src/retrieval.py` khác hash:

| Phiên bản | SHA-256 của `src/retrieval.py` |
| --- | --- |
| Trước | `20977befade4fdd0042e2f820b557fc802563f1297e3571f98ae6788828f4f18` |
| Sau | `f522b29322d235ff50141dd41ae31fac2443c3a9604ffbda44f8622998d6b132` |

Cả hai dùng cùng evaluator hash `260f333f2483e6389feeafba2cd62deedf273c21e7a9f0b3422ef3a620510418`, cùng corpus, cùng E5 và cùng split-manifest hash `04f94354cc188f190244daa056a64a7eba165050336535b24d2b77e51ffcac4d`. Tệp công khai chỉ lưu ID, thứ hạng/chỉ số và loại lỗi; không xuất nội dung câu hỏi hoặc đáp án của các bộ đánh giá.

| Lượt đo | Thời điểm artifact, UTC+7 ngày 23/11/2026 | Mã thoát | Thời gian toàn lệnh |
| --- | --- | ---: | ---: |
| Dev100, bốn phương pháp | 09:26:31 | 0 | 76,826 giây |
| `confirmation100`, hai phương pháp | 09:27:21 | 0 | 50,274 giây |
| Unicode trước sửa, 50 câu × 2 biến thể | 09:32:31 | 0 | 46,131 giây |
| Unicode sau sửa, 50 câu × 2 biến thể | 09:33:23 | 0 | 51,359 giây |

Thời gian toàn lệnh bao gồm khởi tạo và ghi báo cáo; không dùng nó thay cho latency truy xuất từng câu. Hai lượt Unicode không có phép đo latency từng câu, nên không kết luận tăng tốc từ các thời gian này. Trạng thái thành công của bốn lệnh trên không áp dụng cho generation, demo hoặc toàn bộ kiểm thử khác. [Lệnh và mã thoát retrieval](execution.json) · [Lệnh và mã thoát Unicode](robustness/execution.json).

Sau khi chuẩn bị môi trường, dữ liệu và index theo README, có thể đo bản hiện tại vào thư mục kết quả mới:

```bash
python -m benchmarks.retrieval_study final --output .private/retrieval-check
python -m benchmarks.retrieval_study confirm --output .private/retrieval-check
python -m benchmarks.evaluate_unicode_robustness --output .private/unicode-check/after.json
```

Muốn so trước/sau cần dùng đúng source tương ứng và cùng môi trường/corpus; lệnh thứ ba tự nó chỉ đo bản đang chạy. Evaluator Unicode từ chối ghi đè tệp đã tồn tại để giữ nguyên bằng chứng cũ. Không điều chỉnh tiếp trên Dev100, tập xác nhận hoặc 50 biến thể đã xem rồi mô tả kết quả đó như một xác nhận độc lập.
