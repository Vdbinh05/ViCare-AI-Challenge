# ViCare — báo cáo bản cải tiến ngày 23/11/2026

**Thí sinh: Vương Đình Bình.** Báo cáo này ghi kết quả thực thi trên **Linux CPU**. Kết quả Windows trước đó được giữ trong [`results/colab`](../colab/REPORT.md); không trộn hai môi trường để gán chênh lệch cho thuật toán. Việc kiểm chứng notebook được ghi riêng để giữ nguyên hash của bản notebook đã chạy. Các file demo công khai giữ trạng thái, mã tài liệu, trích dẫn và độ trễ; văn bản câu hỏi/câu trả lời và URL từng nguồn được lược khỏi JSON. Nhật ký stage công khai giữ thời gian và mã thoát; đường dẫn nội bộ và dòng lệnh cục bộ được lược. [Phiên Linux có output](../../ViCare_Linux_CPU_Executed.ipynb) · [Phạm vi kiểm chứng notebook](NOTEBOOK_VALIDATION.md).

## 1. Kết quả chính và phạm vi kết luận

Lỗi định tuyến BM25 với dấu câu Unicode đã được sửa. Trên toàn bộ 50 câu NoDiac50 đã khóa, thêm một cặp ngoặc kép cong làm Recall@5 của bản cũ giảm xuống **7/50**; bản sửa đạt **22/50** trên đúng biến thể đó. Đây là mức tăng **30 điểm phần trăm** với **15 câu thắng, 0 câu thua**, trong một phép kiểm tra độ bền đầu vào đã xác định trước lượt đo. Câu nguyên bản giữ **22/50**, và toàn bộ thứ hạng top 10 của 50 câu nguyên bản giữ nguyên trước/sau. [Kết quả và đối chiếu từng câu](RETRIEVAL.md).

Các câu này đã từng dùng chọn cấu hình, nên kết quả **không phải xác nhận trên holdout mới**. Hai biến thể của cùng 50 câu không được tính thành 100 câu độc lập. Sau sửa, tổng Recall@5 của câu nguyên bản và câu bọc ngoặc đều bằng 22/50, nhưng có hai ID đổi hit/miss giữa hai biến thể; chưa đạt bất biến hoàn toàn với dấu câu.

Trên Dev100 chuẩn, Hybrid hiện tại đạt **55/100**, tái lập mức đã công bố trước đó. Trong cùng lượt Linux, Dense đạt **50/100**, Hybrid đạt **55/100**: 7 câu thắng, 2 câu thua ở @5. Đây là so sánh hai phương pháp trong cùng môi trường; không phải mức tăng mới 5 điểm phần trăm do bản vá này. Tập confirmation100 cũng đã được xem trước đợt sửa và không được gọi là holdout chưa từng dùng.

| Phương pháp, cùng phiên Linux | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|---|---:|---:|---:|---:|
| Dense HNSW | 36/100 | 46/100 | 50/100 | 54/100 |
| Hybrid v2 HNSW80 | 37/100 | 51/100 | 55/100 | 59/100 |
| Hybrid HNSW200 đã khóa | 38/100 | 52/100 | 55/100 | 60/100 |
| Hybrid exact hiện tại | 38/100 | 52/100 | 55/100 | 60/100 |

Hybrid exact trên confirmation100 đạt **48/62/68/75** tại @1/3/5/10. Hybrid HNSW200 đạt **49/62/68/75**. Exact không thắng trên mọi tiêu chí. Nguồn: [Dev100](retrieval/dev100.json), [confirmation100](retrieval/confirmation100.json), [robustness trước](robustness/before.json), [robustness sau](robustness/after.json).

## 2. Thay đổi trong mã nguồn

- **Định tuyến truy vấn tiếng Việt:** nhận diện chữ có dấu sau chuẩn hóa NFC; dấu ngoặc cong, emoji và các ký tự Unicode khác không bị coi nhầm là dấu tiếng Việt. Các ca NFD và Đ/đ cũng được kiểm tra.
- **Chọn đoạn đại diện:** giữ đoạn Dense phù hợp khi cùng tài liệu xuất hiện ở cả Dense và BM25. Với tài liệu nhiều đoạn chỉ có từ nhánh sparse, chọn đoạn khớp token truy vấn tốt nhất. Điểm và thứ hạng BM25 theo tài liệu không đổi do bước chọn đoạn này.
- **Trích bằng chứng:** có thể chọn một đến ba câu liền nhau trong cùng nguồn; giữ nguyên văn và khoảng trắng. Bộ kiểm tra phạm vi đọc chính đoạn sẽ hiển thị. Không nối câu ở các nguồn khác nhau hoặc dùng một câu đứng ngoài đoạn hiển thị làm điều kiện ngầm.
- **Kiểm tra bệnh cảnh:** câu hỏi suy ra từ nguồn chỉ là gợi ý có thể sai. Bộ so sánh phải xét câu hỏi thật với đoạn nguồn thật; có thêm kiểm tra lời kể cá nhân/gia đình và tham chiếu ngược thiếu chủ thể.
- **Đo lường:** tách số câu được trả lời, từ chối, cảnh báo cấp cứu và điểm judge theo trạng thái. Bộ kiểm tra demo đòi hỏi quan hệ phép đo–yếu tố nằm trong cùng câu nguồn, tránh ghép các câu không liên quan thành bằng chứng giả.

Corpus chứa văn bản đáp án tham chiếu theo thiết kế known-corpus của BTC. Nhãn ánh xạ gold chỉ dùng trong evaluator. Bản nâng cấp giữ cùng model, số tài liệu hiển thị và ngưỡng cosine. Mã và điều kiện đo: [protocol](PROTOCOL.md), [retrieval](../../src/retrieval.py), [generation](../../src/generation.py), [evaluator Unicode](../../benchmarks/evaluate_unicode_robustness.py), [evaluator generation](../../benchmarks/evaluate_generation.py).

## 3. Generation: không đồng nhất điểm judge với độ chính xác

So sánh dưới đây dùng cùng `dev[:10]`, cùng Linux CPU, cùng model/digest, cùng ngưỡng và timeout **180 giây mỗi yêu cầu** cho cả hai bản. Không dùng kết quả Windows làm baseline của bảng này.

| Chỉ số trên 10 câu | Baseline | Bản cuối |
|---|---:|---:|
| Chatbot trả lời | 3 | 1 |
| Từ chối | 6 | 8 |
| Nhánh cấp cứu | 1 | 1 |
| Legacy Judge chấm 1 | 10 | 10 |
| Câu được trả lời và judge chấm 1 / toàn bộ 10 câu | 3/10 | 1/10 |
| Lỗi kỹ thuật / judge không hợp lệ | 0 / 0 | 0 / 0 |

**Legacy Judge 10/10 không chứng minh chatbot đúng 100%.** Cùng model chấm cả lời từ chối là tốt, và còn bỏ sót một câu baseline có trích dẫn đúng định dạng nhưng sai bệnh cảnh. Chỉ số “câu được trả lời và judge chấm 1” cũng là proxy của model, không phải tỷ lệ trả lời đúng được thẩm định.

Đối chiếu văn bản bằng công cụ AI cho thấy bản cuối đã từ chối ca `d003`, loại được câu trả lời sai ngữ cảnh của baseline. Tuy nhiên `d002` mất một câu trả lời có thông tin được nguồn hỗ trợ một phần; `d000` vẫn chỉ trả lời một phần như baseline. Không tuyên bố đã tăng độ chính xác tổng quát hoặc đã có đánh giá của bác sĩ. Bảng ID và lý do từng ca nằm trong [GENERATION.md](GENERATION.md); số liệu máy ở [baseline180](generation_before180.json), [bản cuối](generation_final.json).

### Thử nghiệm đã loại vẫn được giữ

Lượt span đầu với timeout 45 giây được lưu nguyên trạng trong [generation_after.json](generation_after.json), [demo.json](demo.json) và [nhật ký demo/pytest](execution.json). Lượt đó trả lời một câu sai ngữ cảnh gia đình, legacy Judge 8/10, demo 5/6 do một request CPU quá hạn, và có một test phân biệt “có thể”/“có nên” thất bại. Đây **không phải kết quả của source cuối**.

Sau khi xác định lỗi ở bước so sánh câu hỏi suy ra, bản cuối sửa tiêu chí đối chiếu và kiểm tra lại cả baseline lẫn candidate với cùng timeout 180. Timeout 180 là cấu hình của phép đo CPU và notebook; mặc định 45 giây của ứng dụng không bị sửa. Nhật ký stage và hash source cuối ở [final_execution.json](final_execution.json).

### Kiểm thử và demo của source cuối

**247 test đạt, 0 thất bại, 0 lỗi, 0 bỏ qua**; lượt pytest hoàn tất trong 91,92 giây với hai cờ `VICARE_QA_REAL_INDEX=1` và `VICARE_QA_REAL_OLLAMA=1`. Có kiểm tra E5/Chroma và Ollama thật, ngoài các ca kiểm tra hợp đồng bằng dữ liệu giả lập. Số lượng test theo báo cáo của tác giả. Nhật ký ghi stage pytest cuối trả mã 0; JUnit XML gốc không đi kèm bộ tệp được cung cấp. [Nhật ký pytest](final_execution.json).

**6/6 ca demo đạt**: câu được nguồn hỗ trợ, ngoài phạm vi, thiếu bằng chứng, phân biệt bệnh tăng huyết áp với phép đo, dịch vụ không khả dụng và cấp cứu. Ca lỗi dịch vụ dùng một cổng thử nghiệm được giữ nhưng không lắng nghe, không tắt Ollama thật; thông báo connection-refused tương ứng là lỗi được chủ động tạo trong phép kiểm tra. [Kết quả demo](demo_final.json).

Ca có bằng chứng trả lời nguyên văn từ tài liệu `train-4308`, thẻ `[1]`, về yếu tố ảnh hưởng **kết quả đo lưu huyết não**. Câu ngoài phạm vi “Thủ đô của nước Úc là gì?” được từ chối với `OUT_OF_DOMAIN`; câu ZYX-9000 được từ chối do thiếu bằng chứng. Các ca này không được dùng để suy ra chất lượng trên mọi câu hỏi sức khỏe.

Các stage baseline generation, final generation, demo và pytest đều trả mã thoát **0**. Hash source sau toàn bộ lượt chạy khớp bản đã khóa trước lượt cuối. Đây là bằng chứng kỹ thuật của source này; các hạn chế về câu trả lời ở mục 3 vẫn còn dù mọi test đã đạt.

## 4. Môi trường và khả năng tái lập

- Baseline runtime: commit `b9a586aecf9837fdf991dddd55f07d26b43d3b64`. `src/` và `app.py` ở mốc `af282252ed56190a15d12c17eb258f3bc875f844` giống baseline; các commit xen giữa chủ yếu thêm notebook và tài liệu.
- Linux x86_64, Python 3.11.14, Torch 2.7.1+cpu; E5 và Ollama dùng CPU trong đợt đo này. Toàn bộ phiên bản được ghi ở artifact tương ứng và notebook khi chạy.
- Corpus: **9.895 tài liệu, 11.227 đoạn**; chữ ký corpus `35902972d43e0841495d6f920901c99146d08e7e745cb5670c76d3d9f367ea87`.
- E5-small revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`; Qwen `qwen3:4b-instruct`, digest `0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0`; Ollama 0.40.17.
- Hybrid exact lấy 200 đoạn Dense, giữ 80 trước khi hợp nhất với BM25 80 tài liệu; RRF `k=40`, trọng số 0,5/0,5. Đánh giá top 10 để tính @1/3/5/10; chatbot hiển thị tối đa 5 tài liệu.

Notebook [`ViCare_Vuong_Dinh_Binh.ipynb`](../../ViCare_Vuong_Dinh_Binh.ipynb) chứa snapshot chính source đang dùng, kiểm tra SHA-256 toàn gói và từng file. Không phụ thuộc đường dẫn Windows, GitHub đã đăng nhập hoặc Ollama trên máy tác giả. Mỗi phiên tự ghi runtime, model digest, corpus, hash source, thời gian và kết quả thật; không điền số đo vào output trước khi chạy.

**Chưa xác nhận thực thi trên hosted Colab.** Kiểm chứng notebook Linux và những khác biệt so với Colab được ghi ở [báo cáo kiểm chứng notebook](NOTEBOOK_VALIDATION.md) và [notebook đã chạy trên Linux](../../ViCare_Linux_CPU_Executed.ipynb). Hai sidecar này được loại khỏi snapshot để hash notebook không tự tham chiếu. Notebook mặc định yêu cầu GPU; người chủ động chạy CPU cần đặt `REQUIRE_GPU=False` và nên dùng timeout 180 giây.

Các lệnh đánh giá độc lập, sau khi đã cài đúng môi trường, dựng index và khởi động đúng Ollama:

```bash
python -m benchmarks.retrieval_study final --output results/reproduce/retrieval
python -m benchmarks.retrieval_study confirm --output results/reproduce/retrieval
python -m benchmarks.evaluate_unicode_robustness --output results/reproduce/robustness.json
python -m benchmarks.evaluate_generation --method Hybrid --run-label reproduce --output results/reproduce/generation.json
python -m benchmarks.check_demo --output results/reproduce/demo.json
```

Các evaluator giữ đầy đủ mẫu số và lỗi; dùng thư mục output mới, không ghi đè kết quả đã công bố. Với phép so sánh Unicode trước/sau, chạy cùng evaluator trên hai checkout và cùng index như hướng dẫn trong [RETRIEVAL.md](RETRIEVAL.md). Đáp án chuẩn chỉ dùng ở bước đánh giá, không truyền cho chatbot khi tạo câu trả lời.

## 5. Giới hạn cần giữ trong bài nộp

Recall@5 theo tài liệu không đo đầy đủ chất lượng đoạn trích, tính đúng bệnh cảnh hoặc độ chính xác y khoa. Demo lấy nguyên văn vẫn có thể thiếu điều kiện hoặc lấy nhầm đối tượng. Quy tắc tối đa ba câu và một nguồn khiến khả năng trả lời còn thấp khi cần tổng hợp; ngưỡng cosine 0,86 chưa được hiệu chỉnh lâm sàng. Model sinh và judge giống nhau có thể thiên lệch, còn `dev[:10]` quá nhỏ và đã được xem trong quá trình phát triển.

Dữ liệu ViHealthQA có thể cũ và chưa được thẩm định cho sử dụng lâm sàng. ViCare là demo tra cứu học thuật; không dùng để chẩn đoán, kê đơn hoặc thay thế nhân viên y tế. Việc chạy kỹ thuật thành công không tương đương nghiệm thu của chủ dự án, công khai link nộp bài hay hoàn tất hai bước Facebook/Form của BTC. Các bước nộp và nội dung dự thảo nằm trong [SUBMISSION.md](../../SUBMISSION.md).
