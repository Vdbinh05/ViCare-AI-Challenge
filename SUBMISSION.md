# Bài đăng Facebook

[AI CHALLENGE 24H] - BÀI DỰ THI CỦA: Vương Đình Bình

ViCare AI | Tra cứu kiến thức sức khỏe tiếng Việt có dẫn nguồn

1. Ý tưởng và bài toán

Sau workshop AI Engineer Roadmap cùng Viện AI4LIFE và Rikkei, mình làm ViCare để người đọc có thể mở đúng đoạn tài liệu đứng sau một câu trả lời. Ví dụ, khi hỏi “Những yếu tố nào ảnh hưởng đến kết quả đo lưu huyết não?”, ViCare trích nguồn train-4308 và cho xem đoạn gốc. Khi hỏi “Thủ đô của nước Úc là gì?”, ứng dụng báo câu hỏi nằm ngoài phạm vi sức khỏe.

2. Công cụ và kỹ thuật

Mình dùng ViHealthQA, multilingual-E5-small, Chroma, BM25 và Qwen3 4B Instruct qua Ollama. ViCare kết hợp tìm theo nghĩa và từ khóa, rồi kiểm tra xem đoạn nguồn có trả lời đúng điều người dùng hỏi hay không. Mình cũng sửa lỗi khiến cặp ngoặc kép cong làm giảm khả năng tìm kiếm với câu tiếng Việt không dấu.

3. Kết quả

Trên 100 câu Dev đo cùng máy Linux CPU, Dense tìm được tài liệu chuẩn trong top 5 ở 50 câu và Hybrid ở 55 câu. Trên 50 câu không dấu đã dùng trong phát triển, khi bọc câu hỏi bằng ngoặc kép cong, kết quả tăng từ 7/50 lên 22/50 sau sửa. Với câu không bọc ngoặc, kết quả vẫn là 22/50.

Demo đạt 6/6 ca đã đặt trước. Trên 10 câu đầu Dev, bản cuối trả lời 1 câu, từ chối 8 câu và chuyển hướng cấp cứu 1 câu. Judge tự động chấm 10/10, tính cả tám lời từ chối. Đây vẫn là hạn chế của ViCare. Sản phẩm là demo học thuật, không dùng để tự chẩn đoán hay kê đơn.

THÔNG TIN DÀNH CHO BAN GIÁM KHẢO

Demo có sẵn câu trả lời và nguồn, mở xem ngay:
https://colab.research.google.com/github/Vdbinh05/ViCare-AI-Challenge/blob/ffbc841b5ce371c104b7d6ab4bb16d3c7c8f40f8/ViCare_Linux_CPU_Executed.ipynb#scrollTo=interactive-question

Notebook Google Colab để tự nhập câu hỏi:
https://colab.research.google.com/github/Vdbinh05/ViCare-AI-Challenge/blob/ffbc841b5ce371c104b7d6ab4bb16d3c7c8f40f8/ViCare_Vuong_Dinh_Binh.ipynb

Mã nguồn và báo cáo:
https://github.com/Vdbinh05/ViCare-AI-Challenge

#AIChallenge24h #AI4LIFE #Rikkei

---

## Ảnh đính kèm

Bốn ảnh chụp các ô và output từ phiên notebook Linux CPU. Chọn cả bốn khi đăng.

1. [Ô hỏi đáp, câu trả lời và nguồn train-4308](media/notebook/01-answer.png)
2. [Câu hỏi địa lý ngoài phạm vi](media/notebook/02-out-of-domain.png)
3. [Mã cosine chính xác và hợp nhất thứ hạng](media/notebook/03-source.png)
4. [Bảng truy xuất, điểm Judge và số câu từ chối](media/notebook/04-results.png)

Đăng vào [nhóm Mê IT Nghiện Code](https://www.facebook.com/groups/meitnghiencode), sau đó dùng liên kết bài đăng để gửi [form BTC](https://m4beo6fqhrl.sg.larksuite.com/share/base/form/shrlgB534MeoVcQaAjNI8oP2cBe). Nếu bài chờ duyệt chưa có liên kết, thông báo BTC cho phép dùng liên kết trang cá nhân. Hạn 12:00 ngày 23/11/2026 theo giờ Việt Nam.
