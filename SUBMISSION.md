# Bài đăng dự thi

[AI CHALLENGE 24H] - BÀI DỰ THI CỦA: Vương Đình Bình

Mình làm ViCare, một ứng dụng tra cứu kiến thức sức khỏe tiếng Việt chạy trên máy cá nhân. Khi ViCare trả lời, người đọc có thể mở thẻ nguồn và so câu trả lời với đoạn tài liệu gốc ngay trên cùng màn hình.

**1. Ý tưởng & Bài toán**

Trong tài liệu sức khỏe, hai đoạn có thể cùng nhắc tới “huyết áp” nhưng giải đáp hai câu hỏi khác nhau. Chẳng hạn, yếu tố ảnh hưởng đến một phép đo khác với nguyên nhân của bệnh. Mình muốn ứng dụng kiểm tra cả mối quan hệ mà người dùng thực sự hỏi tới, đồng thời cho người dùng thấy căn cứ của câu trả lời.

**2. Công cụ & Kỹ thuật**

ViCare dùng Python, Gradio, Chroma, multilingual-E5-small, BM25 và Qwen3 4B Instruct chạy bằng Ollama. Câu hỏi được tìm theo nghĩa bằng E5 và theo từ hoặc cụm từ bằng BM25. Sau khi gộp hai bảng xếp hạng theo tài liệu, ViCare kiểm tra phạm vi của bằng chứng, trích nguyên câu từ nguồn được chọn và gắn số để người đọc tra lại. Nếu nguồn chưa đủ, giao diện giải thích lý do từ chối.

**3. Kết quả**

Trên cùng 100 câu dev ViHealthQA, Dense tìm được tài liệu chuẩn trong top 5 ở 51 câu. Hybrid tìm được ở 55 câu, tăng 4 điểm phần trăm. Hybrid cũng chậm hơn trong lần đo này. Với câu hỏi về các yếu tố ảnh hưởng đến kết quả đo lưu huyết não, ứng dụng dẫn đúng đoạn trong nguồn [1]. Câu hỏi về thủ đô nước Úc được nhận diện là ngoài phạm vi.

Trên 10 câu đầu chạy toàn bộ ứng dụng, ViCare trả lời 1 câu có trích dẫn, từ chối 8 câu và chuyển 1 câu sang hướng dẫn cấp cứu. Judge tự động cho 9/10, kể cả 7 lượt từ chối. Báo cáo có đối chiếu từng câu để người đọc tự xem giới hạn này. Đây là bản demo học thuật.

**THÔNG TIN DÀNH CHO BAN GIÁM KHẢO**

- Mã nguồn và hướng dẫn chạy: <https://github.com/Vdbinh05/ViCare-AI-Challenge>
- Bốn ảnh giao diện, kết quả và mã truy xuất: <https://github.com/Vdbinh05/ViCare-AI-Challenge/tree/main/media>
- Số liệu và kết quả từng câu: <https://github.com/Vdbinh05/ViCare-AI-Challenge/blob/main/results/submission/REPORT.md>

#AIChallenge24h #AI4LIFE #Rikkei

---

## Ảnh đính kèm theo thứ tự

| Ảnh | Chú thích đề xuất |
|---|---|
| [01 · Giao diện](media/01-interface.png) | Nhập câu hỏi tiếng Việt và chọn Dense hoặc Hybrid. |
| [02 · Câu trả lời và nguồn](media/02-answer-with-source.png) | Trích dẫn [1] và đoạn nguồn train-4308 cùng nói về các yếu tố ảnh hưởng phép đo lưu huyết não. |
| [03 · Ngoài phạm vi](media/03-out-of-domain.png) | ViCare giải thích vì sao từ chối câu hỏi về thủ đô nước Úc. |
| [04 · Mã truy xuất](media/04-retrieval-source.png) | Mã Python lấy ứng viên Dense và BM25 rồi hợp nhất kết quả ở chế độ Hybrid. |

## Checklist nộp bài

- [ ] Đăng phần từ tiêu đề bài thi đến hết ba hashtag vào [nhóm Mê IT Nghiện Code](https://www.facebook.com/groups/meitnghiencode). Khi dán vào Facebook, dùng các URL hiển thị trong bản văn có thể sao chép và đính kèm bốn ảnh gốc ở trên.
- [ ] Sau khi đăng, sao chép liên kết bài viết và điền [form dự thi của BTC](https://m4beo6fqhrl.sg.larksuite.com/share/base/form/shrlgB534MeoVcQaAjNI8oP2cBe). Bài chờ duyệt vẫn đúng quy trình. Nếu chưa lấy được liên kết bài chờ duyệt, BTC cho phép dùng liên kết trang cá nhân để đối soát.

Hạn theo thông báo: **12:00 trưa, ngày 23/11/2026, giờ Việt Nam (UTC+7)**.
