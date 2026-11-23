# Truy xuất ViCare trên Windows

Hybrid dùng cosine chính xác đạt 125/200 câu đúng tài liệu ở top 5 trên tập chọn cấu hình, so với 123/200 của Hybrid HNSW đã khóa. Trên Dev100, cả hai tìm đúng cùng 55 câu. Trên 100 câu xác nhận mới, cả hai tìm đúng cùng 68 câu, còn exact mất một câu ở top 1. **Chưa xác nhận được mức tăng Recall@5 ngoài tập chọn.** Bản Hybrid v2 cũ vẫn hơn cả hai một câu ở top 5 trên Dev100.

## Cách chọn và khóa tập

- Mỗi câu có một tài liệu chuẩn trong kho ViHealthQA. Recall@K là tỷ lệ câu có mã tài liệu đó trong K mã tài liệu đầu tiên. Đáp án tham chiếu vẫn ở trong corpus theo thiết kế của BTC. Ranker chỉ nhận câu hỏi và dữ liệu trong chỉ mục. Evaluator mới đối chiếu danh sách trả về với nhãn gold.
- Train200 và 50 biến thể không dấu của 50 câu đầu đã tham gia chọn cấu hình. Dev100 cố định đã được dùng phân tích lỗi ở các vòng trước. Confirm100 cũ đã được xem. Các chỉ số Dev100 bên dưới được đo lại sau khi khóa; không dùng chúng để chọn lại. [Quy tắc đã đăng ký](protocol.md)
- Tập xác nhận mới được khóa lúc 08:41:18 ngày 23/11/2026 (UTC+7), trước phép đo retrieval. Dùng seed 2026112302, loại toàn bộ train/validation, 100 câu xác nhận cũ, các câu tổng hợp/trích dẫn đã biết và trùng lặp trong tập mới. Đối chiếu ID nguồn, ID tài liệu sau làm sạch và khử trùng đáp án, bài nguồn theo hostname+path, câu hỏi chuẩn hóa, Jaccard từ ≥0,75 hoặc cụm ba từ ≥0,60. Kiểm tra lại ghi 0 trùng lặp tại từng mức đã nêu. Có thể còn diễn đạt tương đương về nghĩa, lịch sử thử nghiệm ngoài hồ sơ hoặc nội dung từng được mô hình học. Việc loại bài nguồn trùng cũng thay đổi phân bố câu. [Danh sách đã khóa](sets.json) · [Đối chiếu chồng lặp](split_audit.json)
- Tiêu chí công bố trước gồm tăng ít nhất 2 hit @5 trên Train200, mất tối đa 2 hit ở từng @1/@3/@10, mất tối đa 1 hit @5 trên NoDiac50, không có lỗi và p95 dưới 5 giây trên cả hai tập. Exact đạt các tiêu chí, reranker trượt @5 trên cả hai tập. Khóa lựa chọn lúc **08:55:09 UTC+7**, trước phép đo Dev100 và xác nhận mới. Tập xác nhận mới chỉ được đánh giá một lần với baseline đã khóa và cấu hình đã chọn. Sau đó tập này đã được xem. SHA-256 của manifest tập được tính trên UTF-8 với xuống dòng LF, giống nhau giữa Windows và Linux. [Quyết định khóa](selection.json)

## Từ ứng viên đến tài liệu trả về

9.895 tài liệu tạo 11.227 đoạn, tối đa 200 từ, chồng 40 từ. Mọi đoạn và metadata đã đối chiếu với corpus chuẩn, có 0 lỗi ánh xạ. E5 nhận `query: ` hoặc `passage: ` và special tokens. Tokenizer thật ghi trung vị 110, p95 254, tối đa 384 token/đoạn, **0/11.227 đoạn vượt giới hạn 512 token**. Câu hỏi dài nhất trên Train200 là 316 token, Dev100 là 270. Không tạo index chia đoạn mới.

Phép so ghép cặp trên tập chọn dùng **cùng một vector câu hỏi** và toàn bộ vector E5 float32 lấy từ Chroma. Sai lệch chuẩn vector so với 1 lớn nhất là 1,19 × 10⁻⁷. Exact chuẩn hóa lại, tính tích vô hướng với mọi đoạn, phân xử hòa theo ID đoạn. HNSW dùng khoảng cách bình phương L2 trên các vector chuẩn hóa và quy đổi về cosine. Cả hai lấy 200 đoạn, giữ 80 theo cosine rồi ID đoạn. BM25 trả tối đa 80 tài liệu. RRF theo ID tài liệu, k=40, trọng số 0,5 mỗi nhánh và phân xử hòa theo ID tài liệu. Dense độc lập dùng HNSW 100 đoạn, Hybrid v2 dùng HNSW 80 đoạn. Demo top 5 và benchmark top 10 của từng phương pháp sử dụng cùng ngân sách.

| Bộ và thứ hạng trước/sau hợp nhất | Hit@5 | Hit@10 | Hit@20 | Hit@50 |
|---|---:|---:|---:|---:|
| Train200, nhánh Dense HNSW của baseline | 112 | 119 | 133 | 145 |
| Train200, nhánh Dense exact | 113 | 121 | 135 | 148 |
| Train200, nhánh BM25 chung | 107 | 122 | 135 | 143 |
| Train200, RRF HNSW | 123 | 132 | 142 | 155 |
| Train200, RRF exact | 125 | 134 | 142 | 155 |
| Dev100, nhánh Dense exact | 51 | 55 | 59 | 71 |
| Dev100, nhánh BM25 | 48 | 58 | 62 | 70 |
| Dev100, RRF exact | 55 | 60 | 68 | 74 |

Trên Train200, trung bình 4,355 đoạn trong 80 đoạn HNSW lặp mã tài liệu của một đoạn đứng trước. Với exact, con số là 4,23. Trước khi cắt RRF, hợp hai nhánh baseline có gold ở 164/200 câu. Trong 77 câu trượt top 5, **36** nằm ngoài hợp, **9** có trong hợp nhưng ngoài 50 tài liệu đầu, **32** có trong 50 tài liệu đầu. Reranker chỉ có thể tác động trực tiếp tới nhóm sau. Trên Dev100, bản chọn có 18 câu ngoài hợp, 8 câu rơi trước top 50 và 19 câu có gold trong 50 nhưng ngoài top 5. JSONL từng câu giữ hạng gold từng nhánh, hạng RRF, ID đoạn trùng bị bỏ, tập 50 tài liệu và ID các đoạn trả về. [Dev100 từng câu](dev100.jsonl)

Reranker dùng duy nhất `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, revision `1427fd652930e4ba29e8149678df786c240d8825`, từ cache có sẵn. Đầu vào cố định là 50 **tài liệu** đầu sau RRF của baseline và **mọi đoạn** thuộc các tài liệu đó, trung bình 65,02 đoạn/câu Train200. Với mỗi cặp câu hỏi–đoạn, mô hình chạy PyTorch CPU float32, batch 8, tối đa 384 token. Mỗi tài liệu lấy đoạn có logit lớn nhất, hòa theo ID đoạn. Điểm cuối là 0,3/(20+hạng RRF) + 0,7/(20+hạng logit). Cả 250 câu đều chạy đủ, không fallback. Độ trễ rerank là thời gian baseline đã đo cho đúng câu cộng thời gian chấm toàn bộ đoạn cố định. Không ghép reranker với exact và không đổi ngưỡng cosine bằng logit hay điểm RRF. [Mô hình của thử nghiệm](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1)

## Kết quả từng cấu hình

Các cột @K, hợp, C50 đều là **số câu trên N**. Hợp là số câu có gold trong hợp ứng viên trước khi cắt; C50 là số câu có gold trong 50 tài liệu ứng viên. Với Dense, hai cột này dùng riêng nhánh Dense. Thắng/thua so với baseline HNSW đã khóa trên cùng ID ở @5. Độ phủ corpus là N/N và số lỗi truy xuất là 0 cho tất cả các hàng.

| Tập | Cấu hình | @1 | @3 | @5 | @10 | Hợp | C50 | Thắng/thua @5 | TB / trung vị / p95, ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Train200 | HNSW đã khóa | 81 | 108 | 123 | 132 | 164 | 155 | 0/0 | 106,2 / 81,3 / 238,3 |
| Train200 | Exact | 82 | 110 | 125 | 134 | 165 | 155 | 2/0 | 104,0 / 75,8 / 256,2 |
| Train200 | Rerank 50 tài liệu | 87 | 107 | 119 | 130 | 164 | 155 | 8/12 | 2.176,8 / 1.994,8 / 3.734,0 |
| NoDiac50 | HNSW đã khóa | 9 | 20 | 22 | 24 | 35 | 31 | 0/0 | 121,3 / 88,4 / 294,1 |
| NoDiac50 | Exact | 8 | 20 | 22 | 24 | 35 | 31 | 0/0 | 114,1 / 86,2 / 297,2 |
| NoDiac50 | Rerank 50 tài liệu | 9 | 9 | 11 | 18 | 35 | 31 | 0/11 | 1.959,0 / 1.738,7 / 3.118,2 |
| Dev100 | Dense | 37 | 47 | 51 | 55 | 80 | 71 | 2/6 | 24,1 / 21,4 / 40,1 |
| Dev100 | Hybrid v2 | 38 | 52 | 56 | 60 | 83 | 74 | 1/0 | 135,9 / 97,4 / 338,4 |
| Dev100 | HNSW đã khóa | 38 | 52 | 55 | 60 | 82 | 74 | 0/0 | 151,0 / 96,5 / 398,0 |
| Dev100 | Exact được chọn | 38 | 52 | 55 | 60 | 82 | 74 | 0/0 | 139,9 / 91,8 / 342,6 |
| Xác nhận mới 100 | HNSW đã khóa | 49 | 62 | 68 | 75 | 90 | 85 | 0/0 | 99,1 / 84,5 / 227,0 |
| Xác nhận mới 100 | Exact được chọn | 48 | 62 | 68 | 75 | 90 | 85 | 0/0 | 87,5 / 80,1 / 155,0 |

Các kết quả Dev100 của Dense, Hybrid v2 và baseline khóa tái hiện các con số đã báo cáo. So exact với baseline khóa, không có câu nào đổi hit ở @1/3/5/10 trên Dev100. Trên xác nhận mới, không câu nào đổi hit ở @3/5/10; có 0 thắng, 1 thua ở @1. Kết quả chỉ mô tả việc tìm lại tài liệu; chất lượng câu trả lời cần được đo ở bước sinh. [Dev100](dev100.json) · [Xác nhận mới](confirmation100.json) · [Thắng/thua từng câu](paired.json) · [CSV 1.350 lượt](per_question.csv)

## Các câu giải thích chênh lệch

- **t038**, tài liệu `train-314`. HNSW bỏ sót tài liệu trong 80 đoạn, BM25 đứng hạng 5, RRF đứng hạng 18. Exact tìm được tài liệu ở hạng 1 của nhánh dense và hạng 1 sau RRF. **t081**, tài liệu `train-5126`, đi từ hạng RRF 16 lên 5 khi exact đưa tài liệu lên hạng 7 của nhánh dense, còn BM25 giữ hạng 4. Đây là hai thắng @5 trên tập chọn. [Thứ hạng Train200](train200.jsonl)
- **d071**, tài liệu `validation-869`. BM25 đứng hạng 1 ở cả ba Hybrid. Với Hybrid v2, nhánh dense đứng hạng 59 và RRF đứng hạng 5. Với baseline khóa và exact, nhánh dense lần lượt đứng hạng 67, 68 và cùng đưa gold xuống hạng RRF 6. **d003**, tài liệu `validation-975`, vắng cả ở nhánh Dense và BM25 của baseline khóa và exact. [Thứ hạng Dev100](dev100.jsonl)
- **x045**, tài liệu `test-92`. Cả hai vẫn lấy được tài liệu ở top 5. Nhánh dense xuống từ hạng 16 với HNSW còn 17 với exact, BM25 giữ hạng 1. Sau hợp nhất, tài liệu từ hạng 1 xuống hạng 2. Đây là toàn bộ chênh lệch hit trên tập xác nhận; không sửa thuật toán sau khi xem câu này. [Thứ hạng xác nhận](confirmation100.jsonl)

## Môi trường và tái lập

Windows, Python 3.11.0, torch 2.7.1, sentence-transformers 3.4.1, ChromaDB 1.0.20, NumPy 1.26.4. E5 chạy CPU, 4 luồng torch và 1 luồng OpenBLAS ở phép đo này. Khởi tạo và kiểm tra corpus/index trên lượt Dev100 mất 17,63 giây; RSS đỉnh của tiến trình so sánh là 1.379,9 MiB. Ma trận exact chiếm 16,446 MiB; đọc và chuẩn hóa mất 0,247 giây ở lượt chẩn đoán. Tiến trình riêng chấm CE có đỉnh 943,7 MiB, đọc lại danh sách ứng viên cố định. Số này không phải bộ nhớ của một dịch vụ nạp đồng thời CE và toàn bộ retriever. Độ trễ mỗi câu gồm mã hóa câu hỏi và xếp hạng; loại thời gian khởi tạo và ghi báo cáo. Phép đo cuối làm nóng bằng ba câu ngoài tập rồi luân phiên thứ tự phương pháp. Tải hệ thống và phần cứng có thể ảnh hưởng độ trễ. [Số liệu chẩn đoán](train200.json) · [Số liệu rerank](train200_rerank.json)

Từ thư mục dự án, dùng chỉ mục đã dựng bằng `python -m src.retrieval build`. Lệnh dưới đo Dev100 với cấu hình đã khóa trong source trên runtime đang dùng. Chọn thư mục output mới để giữ các kết quả đã lưu.

```shell
python -m benchmarks.retrieval_study final --output .private/retrieval-dev100
```

Ba pha `diagnose`, `rerank` và `select` dùng chung một thư mục output mới nếu cần tái lập thí nghiệm chọn cấu hình. Pha `rerank` yêu cầu snapshot đã có trong cache; demo và benchmark bản chọn không cần reranker. Pha `confirm` tái lập đúng 100 ID đã công bố. Từ lần đo trong báo cáo, tập này đã được xem và không còn là một kiểm tra mù cho các lần chỉnh sửa tiếp theo.
