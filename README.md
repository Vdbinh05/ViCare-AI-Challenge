# ViCare AI

**[Mở bài dự thi của Vương Đình Bình trong Google Colab](https://colab.research.google.com/github/Vdbinh05/ViCare-AI-Challenge/blob/67799e6c09eda71e5d54ec729227afdf737dbd76/ViCare_Vuong_Dinh_Binh.ipynb)** · [Tệp notebook](ViCare_Vuong_Dinh_Binh.ipynb) · [Mã nguồn notebook sử dụng](https://github.com/Vdbinh05/ViCare-AI-Challenge/tree/b9a586aecf9837fdf991dddd55f07d26b43d3b64)

ViCare tra cứu kiến thức sức khỏe tiếng Việt trong ViHealthQA. Người đọc có thể đối chiếu câu trả lời với đoạn nguồn hoặc xem lý do ViCare từ chối. Đây là demo học thuật của AI Challenge 24h. Dữ liệu có thể đã cũ. Chỉ nhập câu hỏi chung, tránh thông tin định danh, chẩn đoán cá nhân và yêu cầu kê đơn.

## Chạy bản nộp trên Colab

1. Mở notebook bằng liên kết trên và kết nối runtime Colab Linux miễn phí. Chọn GPU T4 nếu tài khoản còn hạn mức miễn phí. Ghi nhận thiết bị thực tế ở ô đầu.
2. Chọn **Runtime → Run all**. Notebook lấy mã nguồn ở commit `b9a586a`, cài Python riêng cho ViCare, xây chỉ mục, đo truy xuất, nạp đúng `qwen3:4b-instruct` rồi chấm 10 câu đầu. Các lần chạy lại trong cùng máy ảo dùng được cache. Lần đầu cần mạng tải dữ liệu, thư viện và trọng số.
3. Xem phần **Ví dụ và các đường lỗi**. Trong **Hỏi ViCare**, gõ một câu hỏi chung và chạy lại ô để xem câu trả lời và nguồn trực tiếp. Phần cuối ghi đường dẫn ZIP chứa JSON/CSV. Tải ZIP và notebook có output trước khi ngắt runtime.

Phiên bản notebook `67799e6c09eda71e5d54ec729227afdf737dbd76` dùng source `b9a586aecf9837fdf991dddd55f07d26b43d3b64`. Bộ dữ liệu, embedding, digest Qwen và cấu hình đánh giá đều được ghi trong notebook. [Báo cáo và trạng thái kiểm chứng](results/colab/REPORT.md) · [Ảnh minh chứng](media/final/) · [Bản bài đăng Facebook](SUBMISSION.md)

## Từ câu hỏi đến đoạn dẫn nguồn

Kho tra cứu gồm 9.895 tài liệu từ cột đáp án của ViHealthQA sau khi khử trùng, chia thành 11.227 đoạn, mỗi đoạn tối đa 200 từ và chồng lấn 40 từ. E5 mã hóa đoạn với tiền tố `passage:` và câu hỏi với `query:`. Chroma lưu vector chuẩn hóa 384 chiều. Câu hỏi và nhãn gold của bộ đánh giá không tham gia lập chỉ mục.

- **Dense** dùng tìm kiếm vector. **Hybrid** bổ sung BM25 ở cấp tài liệu, xử lý cụm hai từ và tìm kiếm bỏ dấu khi câu hỏi không có dấu. Dense dùng 100 đoạn HNSW ở cả top 5 và top 10. Nhánh ngữ nghĩa của Hybrid tính cosine chính xác trên ma trận E5 hiện có, lấy 200 đoạn rồi giữ 80 theo cosine và ID đoạn. BM25 lấy 80 tài liệu. Sau khử trùng theo document ID, RRF hợp nhất hai thứ hạng với k=40 và trọng số 0,5 mỗi nhánh.
- Các nhánh cấp cứu và yêu cầu liều dùng cá nhân trả về sớm. Với câu hỏi còn lại, cổng truy xuất dùng cosine Dense tối thiểu 0,86. Đây là ngưỡng minh họa chưa được hiệu chỉnh lâm sàng.
- Ollama kiểm tra liệu nguồn có trả lời **đúng đối tượng, bệnh cảnh và mối quan hệ được hỏi**. ViCare chọn một câu nguyên văn trong nguồn được duyệt, đối chiếu phạm vi và kiểm tra số trích dẫn trước khi hiển thị. Nếu không đủ bằng chứng, ViCare có thể từ chối. Khi Ollama lỗi sau truy xuất, người dùng vẫn đọc được các đoạn đã tìm thấy.

## Cài lần đầu trên Windows

Cần Git, Python 3.11, Ollama và dung lượng trống cho model, cache Hugging Face và chỉ mục. Các phép đo sử dụng E5 trên CPU, RTX 4060 Laptop 8 GB cho Ollama và máy có khoảng 16 GB RAM. Đóng các bản ViCare không còn dùng trước khi chạy đánh giá để dành bộ nhớ cho chỉ mục.

```powershell
git clone https://github.com/Vdbinh05/ViCare-AI-Challenge.git
cd ViCare-AI-Challenge
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Nếu máy mới chưa có model, cài Ollama, mở ứng dụng Ollama và dùng `ollama pull qwen3:4b-instruct` một lần. Trên máy đã có đúng model, dùng lại kho hiện tại. Từ thư mục dự án, tạo chỉ mục một lần bằng lệnh dưới. Lần đầu cần mạng để lấy ViHealthQA và E5 tại revision đã ghim.

```powershell
$env:HF_HOME = Join-Path $PWD '.cache\hf'
.\.venv\Scripts\python.exe -m src.retrieval build
```

## Mở ứng dụng

Trong thư mục clone đã chuẩn bị, chỉ cần chạy:

```powershell
.\.venv\Scripts\python.exe app.py
```

Trên máy dự thi, có thể mở PowerShell ở **bất kỳ thư mục nào** và dùng:

```powershell
& 'D:\TT24H\ViCare-AI-Challenge\.venv\Scripts\python.exe' 'D:\TT24H\ViCare-AI-Challenge\app.py'
```

Mở URL in cạnh PID của phiên mới. ViCare chọn một cổng loopback trong khoảng 7860–7869. Có thể dùng `--port` để yêu cầu một cổng cụ thể. Máy chủ Ollama mặc định ở `http://127.0.0.1:11434`. Ứng dụng kiểm đúng tên model và thực hiện một lượt sinh ngắn. Nếu chưa có daemon, ViCare có thể chạy Ollama đã cài với kho model có sẵn.

| Khi gặp vấn đề | Cách kiểm tra |
|---|---|
| Thiếu chỉ mục hoặc sai phiên bản | Chạy lệnh build ở trên. Nếu dùng chỉ mục riêng, đặt `VICARE_INDEX_DIR`. |
| Không tìm thấy Ollama hoặc model | Kiểm tra `ollama list` trên đúng máy chủ. Đặt `VICARE_OLLAMA_EXE` và `VICARE_OLLAMA_MODELS` khi vị trí cài khác mặc định. Kho `D:\ollama_models` là bố trí máy dự thi. Máy chủ đã chạy dùng kho mà nó nhận lúc khởi động. |
| Câu trả lời báo lỗi dịch vụ | Xem chẩn đoán endpoint, HTTP hoặc timeout ở terminal. Sau khi dịch vụ hoạt động lại, gửi lại câu hỏi trong cùng giao diện. |
| Không mở được phiên mới | Đọc traceback và URL/PID thực tế. Cổng chỉ định bằng `--port` phải trống. Khi Windows báo lỗi nạp DLL hoặc paging file, kiểm tra bộ nhớ còn trống và các phiên Python đang nạp model. |

[Danh mục biến cấu hình](.env.example). Các biến tùy chọn được đọc từ môi trường của tiến trình. Tệp `.env.example` là tài liệu tham khảo.

## Thử trên giao diện

| Câu hỏi tổng hợp | Kết quả cần đối chiếu trên màn hình |
|---|---|
| Những yếu tố nào ảnh hưởng đến kết quả đo lưu huyết não? | Mở nguồn `train-4308` để kiểm tra nguyên câu nói về các yếu tố ảnh hưởng **phép đo**. |
| Thủ đô của nước Úc là gì? | Từ chối với lý do ngoài phạm vi sức khỏe. |
| Thuốc giả định ZYX-9000 có chữa khỏi mọi bệnh mạn tính không? | Từ chối do thiếu bằng chứng. |

## Kết quả đo trên Windows

| Cấu hình, cùng Dev100 | R@1 | R@3 | R@5 | R@10 | Gold trong top 50 ứng viên | Trung vị / p95 |
|---|---:|---:|---:|---:|---:|---:|
| Dense HNSW | 37% | 47% | 51% | 55% | 71% | 21 / 40 ms |
| Hybrid v2, HNSW 80 đoạn | 38% | 52% | 56% | 60% | 74% | 97 / 338 ms |
| Hybrid đã khóa trước đó, HNSW 200 đoạn | 38% | 52% | 55% | 60% | 74% | 96 / 398 ms |
| **Hybrid chọn cho notebook, cosine chính xác** | **38%** | **52%** | **55%** | **60%** | **74%** | **92 / 343 ms** |

Mỗi câu có một tài liệu tham chiếu trong kho tìm kiếm. Ở top 5, bản chọn tìm thêm 6 câu và mất 2 câu so với Dense. Khi thay HNSW trong Hybrid bằng cosine chính xác, Train200 tăng 123→125 hit. NoDiac50 giữ 22/50. Dev100 giữ 55/100 và tập xác nhận mới giữ 68/100. Ở top 1 trên tập xác nhận, bản chọn mất một câu, từ 49 xuống 48/100. Tất cả câu và lần giảm điểm được giữ trong kết quả. [Giao thức](results/colab/retrieval/protocol.md) · [CSV từng câu](results/colab/retrieval/per_question.csv) · [Kết quả Dev100](results/colab/retrieval/dev100.json)

Dev100 đã tham gia phân tích lỗi trong lịch sử. Train200 và NoDiac50 dùng để chọn cấu hình. Confirm100 cũ đã được xem trước đợt này. Tập xác nhận mới được khóa trước phép thử, loại các câu, mã tài liệu và bài nguồn trùng theo quy tắc công bố, và chỉ đánh giá một lần sau khi chọn. [Khóa tập và kiểm tra chồng lặp](results/colab/retrieval/split_audit.json) · [Xác nhận trên 100 câu](results/colab/retrieval/confirmation100.json)

Trên đúng 10 câu đầu dev, cả ba cấu hình đem chấm đều từ chối 9 câu và chuyển cấp cứu 1 câu. Judge cho 9/10 ở mỗi cấu hình, gồm tám lượt từ chối. Agent đối chiếu nguồn và phát hiện ít nhất ba lượt từ chối dù nguồn đầu đã đủ bằng chứng. [Đối chiếu từng câu](results/colab/generation/review.md)

Để tái lập trên Windows, sau khi xây chỉ mục và chuẩn bị Ollama, chạy trong thư mục dự án.

~~~powershell
$env:VICARE_TORCH_THREADS = '4'
$env:OPENBLAS_NUM_THREADS = '1'
.\.venv\Scripts\python.exe -X utf8 -m benchmarks.retrieval_study final --output .private/retrieval-rerun
.\.venv\Scripts\python.exe -X utf8 -m benchmarks.evaluate_generation --method Hybrid --output .private/generation-rerun/final.json
.\.venv\Scripts\python.exe -X utf8 -m benchmarks.check_demo --output .private/demo-rerun.json
~~~

Lệnh generation ghi JSON và CSV rút gọn. Tùy chọn `--save-review-inputs` lưu văn bản để đối chiếu trong `.private/`. [Đánh giá từng câu trả lời](results/colab/generation/review.md) · [Kết quả Windows trước đợt này](results/submission/REPORT.md)

## Nguồn và điều kiện sử dụng

- Thiết kế kho và bộ dev xuất phát từ notebook thực hành của BTC. Mã ứng dụng, kiểm thử và hồ sơ nộp có sử dụng công cụ hỗ trợ lập trình AI. Việc đối chiếu từng câu được ghi rõ trong báo cáo.
- ViHealthQA do Nhung Thi-Hong Nguyen và cộng sự giới thiệu trong *SPBERTQA: A Two-Stage Question Answering System Based on Sentence Transformers for Medical Texts* (KSEM 2022). Tác giả ghi dữ liệu dành cho nghiên cứu và cảnh báo có thể chứa thông tin cá nhân. Repo chỉ phân phối mã tải và xử lý cùng kết quả đã rút gọn. [Dataset card](https://huggingface.co/datasets/tarudesu/ViHealthQA) · [Bài báo](https://arxiv.org/abs/2206.09600)
- Embedding dùng `intfloat/multilingual-e5-small` (MIT). Mô hình sinh là `qwen3:4b-instruct` trong Ollama (Apache-2.0). Trọng số và thư viện phụ thuộc giữ điều kiện sử dụng của nhà phát hành. [E5](https://huggingface.co/intfloat/multilingual-e5-small) · [Qwen trên Ollama](https://ollama.com/library/qwen3:4b-instruct) · [Phiên bản thư viện](requirements.txt)

- Thử nghiệm xếp hạng lại dùng checkpoint mMARCO mMiniLMv2 đã lưu, giấy phép Apache-2.0, revision `1427fd652930e4ba29e8149678df786c240d8825`. Notebook chỉ cần E5 và Qwen cho cấu hình đã chọn. [Model card reranker](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1)
