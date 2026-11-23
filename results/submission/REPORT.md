# Đánh giá ViCare trên ViHealthQA

Ngày đo 23/11/2026 (UTC+7). Hai phép đo trả lời hai câu hỏi khác nhau: bộ truy xuất có tìm lại tài liệu chuẩn, và ứng dụng thực sự trả lời thế nào khi nhận những câu hỏi đó?

## Tìm tài liệu chuẩn

| Phương pháp | R@1 | R@3 | R@5 | R@10 | Coverage | Độ trễ trung bình / trung vị / p95 |
|---|---:|---:|---:|---:|---:|---:|
| Dense | 37/100 | 47/100 | 51/100 | 55/100 | 100/100 | 24,45 / 23,89 / 37,22 ms |
| Hybrid bản nộp | 38/100 | 52/100 | 55/100 | 60/100 | 100/100 | 130,01 / 89,48 / 276,11 ms |

Ở top 5, Hybrid tìm thêm tài liệu chuẩn ở 6 câu, mất ở 2 câu và giữ nguyên kết quả đúng/sai ở 92 câu. Mức tăng 4 điểm phần trăm trên bộ dev này đi cùng chi phí truy xuất cao hơn. Mỗi câu có ít nhất một tài liệu gold trong top K được tính một hit. Coverage đếm câu có gold ở bất kỳ vị trí nào trong chỉ mục. Cả hai mẫu số đều là 100. [CSV 200 lượt theo ID](retrieval/per_question.csv) · [Thông số, môi trường và hash](retrieval/evaluation.json) · [Biểu đồ](retrieval/recall.png)

**Dữ liệu.** Lấy mẫu 100 câu validation theo thứ tự của notebook, seed 2026. Kho gồm các đáp án đã khử trùng trên cả ba split, 9.895 tài liệu, 11.227 đoạn. Kho có chứa tài liệu tham chiếu theo thiết kế của BTC. Kiểm tra lại toàn bộ đoạn trong Chroma với ViHealthQA đã ghim. Dev100 từng được dùng phân tích lỗi nên chỉ cho biết kết quả trên chính tập này.

| Thành phần | Định danh |
|---|---|
| ViHealthQA | `d90a62deedfc822a15de3f35535bb41474e4d3f4` |
| E5-small | `614241f622f53c4eeff9890bdc4f31cfecc418b3` |
| SHA-256 corpus | `35902972d43e0841495d6f920901c99146d08e7e745cb5670c76d3d9f367ea87` |
| SHA-256 danh sách câu hỏi có thứ tự | `5a0f84a1a3f4834af5eb21c2eee5ddb17d977f7fec321fba65e1c5606e3ab8fa` |

**Cách đo.** Chạy ba câu làm nóng cho mỗi phương pháp, sau đó luân phiên phương pháp chạy trước trên cùng câu. Bộ truy xuất chỉ nhận văn bản câu hỏi, phương pháp và K. Gold chỉ dùng để chấm sau khi nhận các document ID. Dense đo top 10 bằng 100 đoạn HNSW. Hybrid lấy 200 đoạn rồi giữ 80 theo cosine/ID, thêm tối đa 80 tài liệu BM25. RRF k=40, hai trọng số 0,5, gộp theo document ID trước khi cắt top K. Nhãn họ thuật toán trong JSON là `Hybrid RRF, Chroma 200, retain 80`; độ sâu HNSW 200 xác định bản nộp. Độ trễ chỉ tính truy xuất. Giao diện lấy 5 nguồn, Dense dùng ngân sách 50 đoạn, Hybrid vẫn dùng 200→80 cùng nhánh BM25. R@5 trong bảng lấy từ năm vị trí đầu của lần đo top 10.

**Vì sao giữ cấu hình này.** Ở vòng phát triển trước, tăng độ sâu HNSW từ 80 lên 200 trước khi cắt còn 80 đã tăng R@5 trên Train200 từ 121 lên 123 và trên 50 câu bỏ dấu từ 21 lên 22. Cả hai tập đã tham gia chọn cấu hình. Khi so hai cấu hình Hybrid trên cùng từng tập khác, Dev100 giảm 56→55/100 và Confirm100 giữ 76→76/100. Không có mức tăng được xác nhận trên Confirm100. Lượt đóng gói giữ lựa chọn đã khóa.

## Đánh giá câu trả lời theo mục 8

| Trên đúng 10 câu đầu | Trước sửa đối chiếu phạm vi | Bản nộp |
|---|---:|---:|
| Trả lời có trích dẫn | 3/10 | 1/10 |
| Từ chối theo trạng thái | 6/10 (60%) | 8/10 (80%) |
| Chuyển hướng cấp cứu | 1/10 | 1/10 |
| Judge@10 | 9/10 | 9/10 |
| Judge không hợp lệ / lỗi kỹ thuật | 0 / 0 | 0 / 0 |

Trong lần chạy đầu, câu `d007` dẫn nguồn khác vị trí giải phẫu, còn `d008` dùng thông tin về khả năng thực hiện để trả lời yêu cầu về khuyến nghị. Bản nộp bổ sung kiểm tra các giới hạn tường minh này ở bước chọn câu nguồn. Giữ nguyên ranking, gold, model, prompt judge và ngưỡng. Ở lần chạy lại, hai ca trên chuyển sang từ chối; danh sách nguồn của cả 10 câu giữ nguyên. Hai câu khác (`d000`, `d004`) vẫn bị từ chối dù nguồn hạng 1 có bằng chứng trực tiếp. Câu được trả lời (`d002`) chỉ giải đáp một phần. Các ví dụ lỗi này đã được xem trong quá trình sửa.

Judge cho 1 điểm ở 7/8 câu bị từ chối và cả lượt chuyển cấp cứu. Vì vậy 9/10 không ước lượng tỷ lệ câu trả lời đúng hoặc độ chính xác y khoa. Phần nhận xét là **đối chiếu từng câu bởi agent**. [Bảng đối chiếu](generation-review.md) · [Số liệu và ID bản nộp](generation.json) · [Mốc trước khi sửa phạm vi](generation-before-scope.json)

## Chạy lại

Sau phần thiết lập trong [README](../../README.md), từ thư mục gốc chạy lần lượt bằng PowerShell. Mỗi lệnh nạp model cần đủ RAM và bộ nhớ ảo trên máy.

~~~powershell
.\.venv\Scripts\python.exe -X utf8 -m src.evaluation --output .private/retrieval-rerun
.\.venv\Scripts\python.exe -X utf8 -m benchmarks.evaluate_generation
$env:VICARE_QA_REAL_INDEX = '1'
$env:VICARE_QA_REAL_OLLAMA = '1'
.\.venv\Scripts\python.exe -X utf8 -m pytest -q --disable-warnings
~~~

Lệnh generation chạy **toàn bộ** ứng dụng ở chế độ Hybrid top 5 trên `dev[:10]`, chấm sau khi ứng dụng trả kết quả, giữ cả ca từ chối/lỗi trong mẫu số. Dùng mặc định ghi trong [.env.example](../../.env.example). Cùng model Ollama `qwen3:4b-instruct` làm model sinh và judge, digest `0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0`. Judge dùng prompt mục 8, đáp án tham chiếu tối đa 1.500 ký tự, temperature 0, tối đa 8 token và chỉ chấp nhận đúng một ký tự 0 hoặc 1. Điểm thiếu/sai định dạng tính 0. Đếm từ chối theo trạng thái ứng dụng, tách riêng cấp cứu và lỗi dịch vụ. Bản ghi đầy đủ để đối chiếu được tạo ở `.private/generation/dev10.json` trên máy người chạy. Các tệp công khai chỉ giữ mã, số đo và nhận xét đã rút gọn.

Notebook gốc lưu Recall@5 50%, Judge@10 7/10, từ chối 3/10, judge không hợp lệ 0. Mô hình judge lịch sử khác model đang có trên máy và không có đủ bản trả lời baseline để chấm lại cùng điều kiện. Vì thế 7/10 chỉ là mốc lịch sử.

## Kiểm chứng ứng dụng

Bộ kiểm thử bản nộp trên bản sao mã từ Git đạt **166/166** khi bật tích hợp thật. Chrome đã chạy đủ 10/10 ca giao diện. Ảnh trình duyệt thật cho thấy câu hỏi về phép đo lưu huyết não dẫn tới nguồn [1] `train-4308`. Đoạn được chọn nói về tác động lên **kết quả đo**. Câu hỏi về thủ đô nước Úc chạy vào nhánh ngoài phạm vi. Trường hợp thuốc giả định ZYX-9000 chạy vào nhánh thiếu bằng chứng sau truy xuất. [Bốn ảnh](../../media/) · [Kiểm tra dịch vụ, cấp cứu và bộ test](verification.md)
