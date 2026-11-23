# Phạm vi và cách đo đợt sửa ViCare ngày 23/11/2026

Ghi nhận lúc **2026-11-23 09:23:00 UTC+7** (**2026-11-23 02:23:00 UTC**).

Đây là biên bản **hồi cố về thay đổi code và cách so sánh**, được viết sau khi lượt generation baseline trên CPU đã hoàn tất. Không coi tài liệu này là đăng ký trước cho lượt chạy đó. Riêng phép đo Unicode ở mục 5 được xác định trước khi chạy cặp đo tương ứng. Tài liệu này không công bố điểm số hoặc kết luận PASS; kết quả và trạng thái thực chạy nằm trong các artifact của đợt đo.

## 1. Mốc source và môi trường

Baseline runtime là commit `b9a586aecf9837fdf991dddd55f07d26b43d3b64`. Runtime trong `src/` và `app.py` của commit `af282252ed56190a15d12c17eb258f3bc875f844` giống mốc này; đã đối chiếu bằng Git. Bản sau sửa được nhận diện bằng commit và hash source thực tế trong báo cáo, không chỉ bằng tên branch.

So sánh generation trước/sau dùng cùng môi trường Linux CPU, Python 3.11.14 và model sinh/judge `qwen3:4b-instruct`. Digest model được giữ và đối chiếu giữa hai lượt:

```text
0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0
```

Đây là kiểm chứng Linux CPU, không phải bằng chứng đã chạy trên hosted Google Colab. Phiên bản thư viện, source hash, corpus hash và thời điểm của từng lượt phải được đọc từ artifact tương ứng.

## 2. Những thành phần giữ cố định

Giữ corpus ViHealthQA và ánh xạ tài liệu/đoạn theo chữ ký của dự án; E5-small revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`, vector chuẩn hóa, chunk 200 từ/overlap 40. Không đổi embedding, thêm model, tune ngưỡng hoặc chọn lại cấu hình từ kết quả đợt đo này.

Các đường truy xuất được phân biệt rõ:

| Đường chạy | Ngân sách |
| --- | --- |
| Dense | HNSW 100 đoạn |
| Hybrid v2 lịch sử | HNSW 80 đoạn, giữ 80 trước hợp nhất |
| Hybrid HNSW đã khóa | HNSW 200 đoạn, giữ 80 trước hợp nhất |
| Hybrid đang chọn | Exact cosine trên vector đã lưu, lấy 200 đoạn rồi giữ 80 trước hợp nhất |

Các đường Hybrid đều dùng BM25 80 tài liệu, RRF `k=40`, trọng số Dense/BM25 `0.5/0.5`. Phép đo lấy top 10 và tính các mốc 1/3/5/10 từ cùng thứ hạng; demo lấy top 5 cùng cấu hình.

Giữ dense gate `0.86`, generation temperature `0` và giới hạn đầu ra cấu hình `384`. Ngưỡng này chưa được hiệu chỉnh lâm sàng. Thay hợp đồng chọn đoạn nguồn để biểu diễn một khoảng câu là sửa pipeline, không phải một vòng tìm model hay ngưỡng mới.

## 3. Phạm vi sửa

- Nhận diện dấu tiếng Việt đúng với NFC/NFD và Đ/đ; dấu câu Unicode, emoji và khoảng trắng đặc biệt không tự chuyển câu không dấu sang nhánh BM25 có dấu.
- Giữ đoạn Dense phù hợp cho tài liệu chung của hai nhánh; với nguồn sparse, chọn đoạn có mức khớp token truy vấn cao nhất, phân xử hòa theo chunk ID. Việc chọn đoạn không đổi điểm/thứ hạng BM25 theo tài liệu, điểm RRF hoặc ngân sách ứng viên.
- Cho phép chọn nguyên văn từ một đến ba câu liền nhau trong cùng nguồn; kiểm tra và hiển thị cùng đoạn đã chọn, giữ kiểm tra phạm vi, điều kiện và trích dẫn.
- Kiểm tra demo tách câu trước khi chuẩn hóa, tránh ghép các từ ở những câu không liên quan thành bằng chứng giả về quan hệ.

Đây là các sửa đổi tổng quát, không hardcode đáp án theo ID benchmark. Sửa định tuyến Unicode có thể thay đổi thứ hạng của đầu vào bị ảnh hưởng. Sửa đoạn bằng chứng có thể thay đổi câu trả lời dù Recall theo tài liệu giữ nguyên; vì vậy vẫn phải đo lại generation.

## 4. So sánh retrieval và generation

Đo lại Dev100 chuẩn và tập `confirmation100` lưu trong manifest khóa. Cả hai đã được xem hoặc công khai trước đợt sửa này; kết quả là **tái lập và so sánh trên tập đã biết**, không phải xác nhận độc lập trên holdout chưa từng xem. Giữ toàn bộ tài liệu trong corpus của bài toán known-corpus retrieval.

Generation trước/sau dùng đúng `dev[:10]`, cùng judge, prompt judge, giới hạn gold 1.500 ký tự và judge options `temperature=0, num_predict=8`. Mỗi câu đi qua truy xuất và pipeline thực của phiên bản tương ứng. Giữ đủ mẫu số, báo lỗi và judge không hợp lệ; tách trạng thái trả lời, từ chối, cấp cứu và không sẵn sàng.

Judge đánh giá tích cực một lời từ chối không được diễn giải thành một câu trả lời kiến thức đúng. Báo khả năng trả lời và độ hỗ trợ của nguồn cùng với Judge@10. Judge cùng họ/model với generator và mẫu nhỏ là các giới hạn của phép đo. Không coi đối chiếu bởi agent là thẩm định của bác sĩ hay nghiệm thu của chủ dự án.

## 5. Phép đo Unicode được xác định trước khi thực chạy

Dùng `benchmarks/evaluate_unicode_robustness.py` giống hệt nhau trong checkout baseline và bản sau sửa. Với **toàn bộ 50 ID của NoDiac50 đã từng dùng chọn cấu hình**, tạo đúng hai đầu vào:

1. `original`: giữ nguyên câu không dấu do loader NoDiac50 tái tạo từ 50 ID đầu của Train200 đã khóa.
2. `curly_quotes`: thêm duy nhất U+201C ở đầu và U+201D ở cuối câu đó.

Đây là **50 câu gốc, hai biến thể mỗi câu**, không phải 100 câu độc lập hay một tập holdout mới. Gọi production retriever với backend `exact`, phương pháp `Hybrid`, `k=10` và các ngân sách ở mục 2.

Mỗi biến thể báo Recall@1/3/5/10 với mẫu số 50; lỗi được tính là miss. Lưu ID câu, gold/retrieved document IDs và loại lỗi, không lưu nội dung câu hỏi/đáp án. Ghi source hash, phiên bản, cấu hình, corpus hash và split-manifest hash để kiểm tra tính đồng nhất.

Không chọn thêm biến thể, chỉnh trọng số/ngưỡng hoặc tune tiếp theo đầu ra của phép đo này. Mọi mức thay đổi chỉ được mô tả là độ bền trước dấu ngoặc kép trên những ví dụ đã dùng trong phát triển, không suy rộng thành mức tăng Recall độc lập trên câu hỏi mới.

## 6. Công bố theo bằng chứng

Không trộn số đo Windows cũ với Linux mới để gán chênh lệch cho thuật toán. Không thay kết quả chưa chạy bằng số dự kiến. Nếu một lượt lỗi hoặc thiếu mẫu, giữ nguyên trạng thái và giải thích giới hạn; không tuyên bố hoàn tất trước khi có artifact đủ và đối chiếu được.
