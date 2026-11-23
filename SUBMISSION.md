# Bài đăng Facebook

[AI Challenge 24h] - Vương Đình Bình

[AI CHALLENGE 24H] - BÀI DỰ THI CỦA: Vương Đình Bình

Sau workshop AI Engineer Roadmap cùng Viện AI4LIFE - ĐH Bách Khoa Hà Nội & Rikkei, mình đã hoàn thiện ViCare AI, trợ lý tra cứu kiến thức sức khỏe tiếng Việt có dẫn nguồn.

1. Ý tưởng & Bài toán giải quyết:

- Bài toán: Khi đọc một câu trả lời về sức khỏe, làm sao biết tài liệu gốc có thực sự trả lời đúng câu mình hỏi?
- Giải pháp: ViCare tìm trong bộ ViHealthQA theo cả ý nghĩa và từ khóa, rồi đối chiếu bằng chứng trước khi trả lời. Người đọc có thể mở bài nguồn để kiểm tra. Khi thiếu căn cứ hoặc gặp câu hỏi ngoài phạm vi, ViCare giải thích lý do từ chối.

2. Công cụ & Kỹ thuật sử dụng:

- Framework/Thư viện: Python, PyTorch, Sentence Transformers, Chroma và Gradio. Mình dùng BM25 để tìm theo từ khóa và RRF để kết hợp các thứ hạng.
- Mô hình: multilingual-E5-small để tìm đoạn liên quan, Qwen3 4B Instruct chạy qua Ollama để kiểm tra bằng chứng và tạo câu trả lời.

3. Kết quả đạt được:

- Với câu “Những yếu tố nào ảnh hưởng đến kết quả đo lưu huyết não?”, ViCare trích đoạn từ tài liệu train-4308 và hiển thị liên kết bài gốc. Với câu “Thủ đô của nước Úc là gì?”, ứng dụng nhận diện đúng là ngoài phạm vi sức khỏe.
- Trên cùng 100 câu đánh giá ở Linux CPU, tìm kiếm theo ngữ nghĩa đưa tài liệu đúng vào top 5 ở 50 câu. Khi kết hợp thêm từ khóa, kết quả là 55 câu.
- ViCare đạt 6/6 ca demo đã định trước. Trên 10 câu đầu của bộ đánh giá, hệ thống trả lời 1 câu, từ chối 8 câu và chuyển hướng cấp cứu 1 câu. Giảm các lượt từ chối khi nguồn đã đủ bằng chứng là phần mình muốn cải thiện tiếp.

THÔNG TIN DÀNH CHO BAN GIÁM KHẢO:

- Link Source Code (GitHub công khai):
https://github.com/Vdbinh05/ViCare-AI-Challenge

- Link chạy Web Demo (Gradio trong Google Colab):
https://colab.research.google.com/github/Vdbinh05/ViCare-AI-Challenge/blob/8e668bf53932c57734c0366569ecf14d7f7d835f/ViCare_Web_Demo.ipynb
Đăng nhập Google, chọn Runtime → Run all, rồi nhập câu hỏi ở ô cuối. Lần đầu cần tải mô hình và dựng chỉ mục.

- Link xem ngay câu trả lời và nguồn từ phiên đã chạy:
https://colab.research.google.com/github/Vdbinh05/ViCare-AI-Challenge/blob/ffbc841b5ce371c104b7d6ab4bb16d3c7c8f40f8/ViCare_Linux_CPU_Executed.ipynb#scrollTo=interactive-question

Mình đính kèm ảnh giao diện, kết quả thực tế, mã nguồn và hướng dẫn chạy. ViCare là demo học thuật, thông tin chỉ để tham khảo.

#AIChallenge24h #AI4LIFE #Rikkei

---

## Ảnh đính kèm theo thứ tự

1. [Giao diện hỏi đáp Gradio, chụp trên Windows](media/facebook/01-giao-dien.png)
2. [Câu trả lời thực tế và tài liệu train-4308, từ phiên notebook Linux CPU](media/facebook/02-ket-qua.png)
3. [Mã nguồn tìm kiếm cosine và hợp nhất thứ hạng](media/facebook/03-ma-nguon.png)
4. [Câu hỏi ngoài lĩnh vực và phản hồi thực tế](media/facebook/04-ngoai-pham-vi.png)
5. [Hướng dẫn chạy trong notebook dành cho giám khảo](media/facebook/05-huong-dan.png)

Đăng vào [nhóm Mê IT Nghiện Code](https://www.facebook.com/groups/meitnghiencode), sau đó dùng liên kết bài đăng để gửi [form BTC](https://m4beo6fqhrl.sg.larksuite.com/share/base/form/shrlgB534MeoVcQaAjNI8oP2cBe).
