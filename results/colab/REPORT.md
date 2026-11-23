# Đánh giá ViCare cho bài nộp Colab

Ngày 23/11/2026, giờ Việt Nam. Mã nguồn tính điểm và chạy demo là [b9a586a](https://github.com/Vdbinh05/ViCare-AI-Challenge/tree/b9a586aecf9837fdf991dddd55f07d26b43d3b64). [Notebook trên Colab](https://colab.research.google.com/github/Vdbinh05/ViCare-AI-Challenge/blob/67799e6c09eda71e5d54ec729227afdf737dbd76/ViCare_Vuong_Dinh_Binh.ipynb)

## Cấu hình đã chọn

ViCare giữ nguyên kho 9.895 tài liệu, 11.227 đoạn, embedding E5 và mô hình sinh `qwen3:4b-instruct`. Trong Hybrid, phép tìm cosine chính xác đọc lại ma trận vector từ Chroma. Cả bản HNSW đã khóa và bản cosine chính xác lấy 200 đoạn, giữ 80 đoạn với cùng cách xử lý đồng điểm, lấy 80 tài liệu BM25 rồi hợp nhất bằng RRF. Phần trả lời sử dụng năm tài liệu đầu, ngưỡng cosine 0,86 và các bước kiểm tra bằng chứng của ứng dụng. Dense dùng cùng ngân sách 100 đoạn khi trả năm và mười nguồn. Trước đợt này, demo Dense lấy 50 đoạn khi trả năm nguồn.

| Tập chọn, số hit @5 | Hybrid HNSW | Cosine chính xác | Reranker mMARCO |
|---|---:|---:|---:|
| Train200 | 123/200 | **125/200** | 119/200 |
| NoDiac50 | 22/50 | **22/50** | 11/50 |

Hai câu Train200 được phục hồi là `t038` (nguồn `train-314`, hạng 18→1) và `t081` (`train-5126`, 16→5). Reranker chấm tất cả đoạn của 50 tài liệu ứng viên đã khóa bằng checkpoint mMARCO có sẵn trên máy. Trên Train200, p95 của phương án này khoảng 3,73 giây và @5 giảm 4 hit so với Hybrid HNSW. Vì vậy cấu hình nộp chỉ cần E5 và Qwen. Tokenizer E5 đếm cả tiền tố và special token, cho trung vị 110, p95 254, tối đa 384 token trên 11.227 đoạn. Số đoạn vượt giới hạn 512 là 0. Ma trận exact dùng 16,45 MiB và mất 0,25 giây để nạp trong lượt chẩn đoán Windows. [Giao thức](retrieval/protocol.md) · [Quyết định khóa](retrieval/selection.json) · [Train200](retrieval/train200.json) · [Reranker](retrieval/train200_rerank.json)

## Truy xuất thực tế trên Windows

| Tập và cấu hình | R@1 | R@3 | R@5 | R@10 | Trong index | Hợp ứng viên | Trong 50 tài liệu | Trung vị / p95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dev100 · Dense | 37% | 47% | 51% | 55% | 100% | 80% | 71% | 21 / 40 |
| Dev100 · Hybrid v2 (80 đoạn HNSW) | 38% | 52% | 56% | 60% | 100% | 83% | 74% | 97 / 338 |
| Dev100 · Hybrid đã khóa (200 đoạn HNSW) | 38% | 52% | 55% | 60% | 100% | 82% | 74% | 96 / 398 |
| Dev100 · Hybrid chọn (cosine chính xác) | 38% | 52% | 55% | 60% | 100% | 82% | 74% | 92 / 343 |
| Xác nhận mới · Hybrid đã khóa | 49% | 62% | 68% | 75% | 100% | 90% | 85% | 85 / 227 |
| Xác nhận mới · Hybrid chọn | 48% | 62% | 68% | 75% | 100% | 90% | 85% | 80 / 155 |

Recall@K đếm câu có ít nhất một tài liệu chuẩn trong K tài liệu đầu. Với Dense, cột hợp ứng viên chỉ có một nhánh. Với Hybrid chọn, 50 tài liệu đầu bao gồm trung bình 63,77 đoạn trên Dev100. Trong 45 lượt trượt @5, 18 lượt thiếu gold ở cả hai nhánh, 8 lượt mất gold khi cắt xuống 50 tài liệu và 19 lượt còn gold trong 50 tài liệu. Thời gian đo có E5 và xếp hạng sau làm nóng, tách khỏi khởi tạo và gọi Qwen. Đây là số đo trên một máy. [Dữ liệu từng câu](retrieval/per_question.csv) · [Tổng hợp Dev100](retrieval/dev100.json) · [Xác nhận mới](retrieval/confirmation100.json)

Ở Dev100, câu `d005` có tài liệu `validation-663` từ hạng 7 của Dense lên hạng 2 của Hybrid chọn. Ở `d007`, tài liệu `validation-976` rơi từ hạng 3 xuống hạng 6. Khi chỉ thay HNSW bằng exact trong Hybrid, toàn bộ hit/miss trên Dev100 tại @1/3/5/10 giữ nguyên. Hybrid v2 giữ `validation-869` ở hạng 5 của `d071`, còn bản chọn xếp hạng 6. Trên tập xác nhận, `x045` (`test-92`) từ hạng 1 xuống 2. Mức tăng @5 trên Train200 chưa được xác nhận ở hai tập báo cáo. [Thắng/thua theo ID](retrieval/paired.json)

Tập 100 câu xác nhận mới được khóa lúc 08:41 trước thí nghiệm. Đối chiếu trực tiếp với 8.102 hàng được bảo vệ (toàn bộ train, validation và Confirm100 cũ) cho 0 câu trùng chính xác sau chuẩn hóa, 0 tài liệu chuẩn trùng, 0 bài nguồn trùng. Tập mới cũng không tự trùng ba khóa đó. Bộ lọc từ vựng được mô tả trong hồ sơ có thể bỏ sót các diễn đạt khác từ vựng. Tất cả tài liệu chuẩn vẫn có trong corpus theo thiết kế bài tập. Dev100 đã dùng phân tích lỗi, Train200/NoDiac50 dùng chọn cấu hình, và cả hai tập xác nhận hiện đã được xem. [Danh sách khóa](retrieval/sets.json) · [Kiểm tra chồng lặp](retrieval/split_audit.json)


## Chất lượng câu trả lời trên Windows

Trên đúng `dev[:10]`, cả Dense, Hybrid HNSW và Hybrid cosine chính xác đều có **0 trả lời có trích dẫn, 9 từ chối, 1 chuyển cấp cứu**. Judge cho 9/10 ở mỗi cấu hình, gồm tám lượt từ chối và một lượt cấp cứu. Cả ba lượt có 0 phiếu chấm sai định dạng, 0 lỗi gọi judge và 0 lỗi kỹ thuật. Hai cấu hình Hybrid trả cùng nội dung trên cả mười câu. Ở bản chọn, độ trễ trả lời trung vị/p95 là 2,21/3,07 giây, chưa tính judge. Bản đo Windows cũ có 1 câu trả lời một phần, 8 từ chối, 1 cấp cứu. Lượt mới không xác nhận chất lượng trả lời tăng.

Đối chiếu bởi agent trên 135 lượt đoạn nguồn cho thấy `d000`, `d002` và `d004` đều có gold hạng 1 và cổng bằng chứng đã chấp nhận đúng nguồn, nhưng ứng dụng vẫn từ chối. Ví dụ `d004` có mô tả trực tiếp đặc tính dùng một lần của dụng cụ. Chất lượng bị giới hạn ở bước sau truy xuất. Phạm vi ít câu và việc dùng cùng Qwen để sinh lẫn chấm khiến điểm Judge khó diễn giải. [Đối chiếu từng câu](generation/review.md) · [Từng lượt nguồn](generation/source-review.csv) · [Bản chọn](generation/selected.json)

## Kiểm chứng ứng dụng Windows

- Bật cả hai cờ tích hợp E5/Chroma/Ollama và chạy toàn bộ pytest trên mã dùng cho notebook. **172 kiểm thử đạt**, thời gian 29,54 giây.
- Sáu ca tổng hợp đạt. Câu kiểm tra phép đo có câu trả lời và nguồn `train-4308`. Câu về thủ đô nước Úc đi qua nhánh ngoài lĩnh vực. Câu về thuốc giả định ZYX-9000 và câu hỏi về nguyên nhân tăng huyết áp đi qua nhánh thiếu bằng chứng. Khi chỉ chuyển endpoint của ca thử sang cổng đóng, ứng dụng báo lỗi dịch vụ và giữ nguồn. Câu cấp cứu vẫn chuyển hướng 115.
- Mở `app.py` từ ngoài thư mục dự án. Chrome gửi năm câu qua giao diện của chính tiến trình mới. Bốn ảnh chụp gồm giao diện, câu trả lời kèm đoạn trích, ca địa lý và mã Python thực tế. [Số liệu kiểm chứng](windows-verification.json) · [Kết quả các ca tổng hợp](generation/demo.json) · [Ảnh](../../media/final/)

## Kiểm chứng Colab

Runtime Google Colab miễn phí bắt đầu lúc 09:17 ngày 23/11/2026 (giờ Việt Nam), dùng 2 luồng CPU, RAM 12,67 GiB và còn 107,62 GiB đĩa lúc khởi động. Kernel dùng Python 3.13.18. Môi trường riêng của ViCare dùng Python 3.11.14 và Torch 2.7.1+cpu. Lệnh pytest mặc định đạt 167 ca và bỏ qua 5 ca tích hợp theo cờ cấu hình. Một lượt WSL Ubuntu độc lập cũng đạt 167 ca và bỏ qua cùng 5 ca khi dùng thư mục tạm Linux.

Colab lấy đúng mã nguồn `b9a586a` và mất 2.115,58 giây để dựng chỉ mục E5. Corpus có 9.895 tài liệu, 11.227 đoạn. Chữ ký corpus và chữ ký 100 câu theo thứ tự khớp với lượt Windows. Bốn cấu hình dưới đây được đo lại trên cùng máy ảo Google, cùng kho và cùng Dev100.

| Cấu hình trên Colab | R@1 | R@3 | R@5 | R@10 | Trong index | Hợp ứng viên | Trong 50 tài liệu | Trung vị / p95 (ms) | Lỗi |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense | 37% | 47% | 51% | 55% | 100% | 79% | 71% | 57 / 308 | 0 |
| Hybrid v2 | 38% | 51% | 54% | 59% | 100% | 83% | 74% | 277 / 938 | 0 |
| Hybrid HNSW đã khóa | 38% | 52% | 55% | 60% | 100% | 82% | 74% | 297 / 1.092 | 0 |
| Hybrid cosine chính xác | 38% | 52% | 55% | 60% | 100% | 82% | 74% | 261 / 1.094 | 0 |

Bản chọn tìm được thêm 4 tài liệu chuẩn trong top 5 so với Dense trên cả hai máy. So với Hybrid đã khóa, số hit tại cả bốn mức bằng nhau ở từng môi trường. Hybrid v2 đạt 54/100 ở top 5 trên Colab và 56/100 trên Windows. Hai máy dựng chỉ mục riêng. Chênh lệch giữa hai lượt đo chưa xác định được nguyên nhân.

Ollama 0.40.17 khởi động trong máy ảo và trả lời câu thử bằng `OK`. Tag `qwen3:4b-instruct` có digest `0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0`. Ollama báo 3.186.108.660 byte bộ nhớ cho mô hình đã nạp và 0 byte VRAM. Lượt đánh giá câu trả lời trên Colab bắt đầu lúc 09:59. Tại mốc công bố kết quả truy xuất, ô này đang chạy. Các số Judge và từ chối ở phần Windows được giữ riêng.
