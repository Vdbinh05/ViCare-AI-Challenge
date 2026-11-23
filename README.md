# ViCare AI

**[Mở demo có sẵn câu trả lời và nguồn](https://colab.research.google.com/github/Vdbinh05/ViCare-AI-Challenge/blob/ffbc841b5ce371c104b7d6ab4bb16d3c7c8f40f8/ViCare_Linux_CPU_Executed.ipynb#scrollTo=interactive-question)** · **[Chạy notebook trên Google Colab](https://colab.research.google.com/github/Vdbinh05/ViCare-AI-Challenge/blob/ffbc841b5ce371c104b7d6ab4bb16d3c7c8f40f8/ViCare_Vuong_Dinh_Binh.ipynb)**

Bài dự thi AI Challenge 24h của Vương Đình Bình. ViCare tìm tài liệu sức khỏe tiếng Việt trong ViHealthQA, kiểm tra bằng chứng và hiển thị đoạn nguồn. Người xem có thể mở demo ngay để đọc câu trả lời, nguồn trích và các lượt từ chối. [Bốn ảnh minh chứng](media/notebook/) · [Bài đăng dự thi](SUBMISSION.md)

## Chạy notebook

1. Mở notebook sạch qua liên kết Google Colab phía trên. Bản mặc định yêu cầu GPU. Chọn T4 nếu tài khoản còn hạn mức miễn phí. Để chạy trên CPU, bỏ chọn `REQUIRE_GPU` ngay ô cấu hình.
2. Chọn **Runtime → Run all**. Notebook khôi phục snapshot source đã kiểm SHA-256, thiết lập môi trường, xây index E5, khởi động Ollama với `qwen3:4b-instruct` rồi chạy các bảng đánh giá. Máy mới cần mạng và thời gian tải dữ liệu, thư viện cùng trọng số.
3. Ở mục **Nhập câu hỏi và kiểm tra nguồn**, thay nội dung câu hỏi và chạy lại ô **Hỏi ViCare**. Các ô cuối xuất JSON/CSV của chính phiên chạy.

Bản đính kèm có output ghi 15/15 ô thực thi qua IPython trên Linux CPU với cache có sẵn. Bản snapshot này chưa có lượt chạy kiểm chứng trên máy ảo Google. Notebook sạch và notebook có output cùng chứa 58 tệp với SHA-256 gói `21545d22b4ea5816d07775e663414472ceef0ae7cb2bce4ecde11f62195f60f5`. [Mã nguồn đúng phiên bản](https://github.com/Vdbinh05/ViCare-AI-Challenge/tree/ffbc841b5ce371c104b7d6ab4bb16d3c7c8f40f8) · [Biên bản Linux](results/upgrade_20261123/NOTEBOOK_VALIDATION.md)

## Kết quả bản nộp

| Phương pháp trên cùng Dev100, Linux CPU | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|---|---:|---:|---:|---:|
| Dense HNSW | 36% | 46% | 50% | 54% |
| Hybrid v2 | 37% | 51% | 55% | 59% |
| Hybrid HNSW đã khóa | 38% | 52% | 55% | 60% |
| Hybrid cosine chính xác | 38% | 52% | 55% | 60% |

Trên 50 câu không dấu đã dùng trong phát triển, khi thêm ngoặc kép cong, Recall@5 tăng từ 7/50 lên 22/50 sau sửa. Câu không bọc ngoặc giữ 22/50. Ở đúng 10 câu đầu Dev, bản cuối trả lời 1 câu, từ chối 8 và chuyển hướng cấp cứu 1. Judge ghi 10/10, gồm cả tám lời từ chối. Bộ demo tổng hợp đạt 6/6. Đây là demo học thuật, không dùng để chẩn đoán hoặc kê đơn. [Báo cáo đầy đủ](results/upgrade_20261123/REPORT.md) · [Đối chiếu từng câu](results/upgrade_20261123/GENERATION.md) · [Đánh giá Unicode](results/upgrade_20261123/RETRIEVAL.md)

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

[Số liệu phiên bản Windows và Colab trước ngày bàn giao](results/colab/REPORT.md)

## Nguồn và điều kiện sử dụng

- Thiết kế kho và bộ dev xuất phát từ notebook thực hành của BTC. Mã ứng dụng, kiểm thử và hồ sơ nộp có sử dụng công cụ hỗ trợ lập trình AI. Việc đối chiếu từng câu được ghi rõ trong báo cáo.
- ViHealthQA do Nhung Thi-Hong Nguyen và cộng sự giới thiệu trong *SPBERTQA: A Two-Stage Question Answering System Based on Sentence Transformers for Medical Texts* (KSEM 2022). Tác giả ghi dữ liệu dành cho nghiên cứu và cảnh báo có thể chứa thông tin cá nhân. Repo chỉ phân phối mã tải và xử lý cùng kết quả đã rút gọn. [Dataset card](https://huggingface.co/datasets/tarudesu/ViHealthQA) · [Bài báo](https://arxiv.org/abs/2206.09600)
- Embedding dùng `intfloat/multilingual-e5-small` (MIT). Mô hình sinh là `qwen3:4b-instruct` trong Ollama (Apache-2.0). Trọng số và thư viện phụ thuộc giữ điều kiện sử dụng của nhà phát hành. [E5](https://huggingface.co/intfloat/multilingual-e5-small) · [Qwen trên Ollama](https://ollama.com/library/qwen3:4b-instruct) · [Phiên bản thư viện](requirements.txt)

- Thử nghiệm xếp hạng lại dùng checkpoint mMARCO mMiniLMv2 đã lưu, giấy phép Apache-2.0, revision `1427fd652930e4ba29e8149678df786c240d8825`. Notebook chỉ cần E5 và Qwen cho cấu hình đã chọn. [Model card reranker](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1)
