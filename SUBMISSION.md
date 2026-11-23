# Bài đăng dự thi

[AI CHALLENGE 24H] - BÀI DỰ THI CỦA: Vương Đình Bình

Mình làm ViCare để người đọc có thể kiểm tra một câu trả lời về sức khỏe ngay tại đoạn tài liệu được dẫn. Bản nộp chạy trong Google Colab. Người xem có thể nhập câu hỏi tiếng Việt, xem nguồn, rồi tự chạy lại các bảng đánh giá.

**1. Ý tưởng & Bài toán**

Hai đoạn cùng có chữ “huyết áp” vẫn có thể trả lời hai vấn đề khác nhau. Một đoạn nói huyết áp ảnh hưởng đến *kết quả phép đo* không đủ căn cứ để giải thích nguyên nhân *tăng huyết áp*. ViCare kiểm tra câu trong nguồn có đúng đối tượng và mối quan hệ người dùng hỏi hay không.

**2. Công cụ & Kỹ thuật**

Mình dùng ViHealthQA, multilingual-E5-small, Chroma, BM25 và Qwen3 4B Instruct chạy qua Ollama. Bản Hybrid kết hợp tìm theo nghĩa với tìm theo từ. Trong đợt này, mình so tìm kiếm HNSW với tính cosine chính xác trên cùng vector, và thử xếp hạng lại 50 tài liệu bằng mMARCO. Bản Colab dùng cosine chính xác và BM25. Tài liệu tìm được đi qua bước kiểm tra bằng chứng trước khi ViCare trả lời có trích dẫn hoặc từ chối.

**3. Kết quả**

Trên cùng Dev100 đo ở Windows, Dense tìm được tài liệu chuẩn trong top 5 ở 51/100 câu. Hybrid đã khóa trước đó và Hybrid chọn cho notebook cùng đạt 55/100. Hybrid v2 từng dùng 80 đoạn HNSW đạt 56/100 khi đo lại. Cosine chính xác tăng từ 123 lên 125 hit trên 200 câu chọn cấu hình, nhưng tập xác nhận bổ sung vẫn giữ 68/100. Mình công bố cả thử nghiệm reranker bị giảm điểm.

Phần đánh giá trả lời dùng đúng 10 câu đầu dev. ViCare từ chối 9 câu và chuyển hướng cấp cứu 1 câu. Judge tự động vẫn cho 9/10, kể cả tám câu từ chối. Ở câu hỏi demo riêng về yếu tố ảnh hưởng đến kết quả đo lưu huyết não, ViCare trích đúng nguồn `train-4308`. Câu hỏi địa lý được nhận diện là ngoài phạm vi. Đây là demo học thuật, không dùng để tự chẩn đoán hoặc kê đơn.

**THÔNG TIN DÀNH CHO BAN GIÁM KHẢO**

- Notebook chính trên Google Colab: https://colab.research.google.com/github/Vdbinh05/ViCare-AI-Challenge/blob/67799e6c09eda71e5d54ec729227afdf737dbd76/ViCare_Vuong_Dinh_Binh.ipynb
- Mã nguồn notebook sử dụng: https://github.com/Vdbinh05/ViCare-AI-Challenge/tree/b9a586aecf9837fdf991dddd55f07d26b43d3b64
- Báo cáo, kết quả từng câu và cách đọc Judge: https://github.com/Vdbinh05/ViCare-AI-Challenge/blob/main/results/colab/REPORT.md
- Ảnh giao diện, kết quả và mã nguồn: https://github.com/Vdbinh05/ViCare-AI-Challenge/tree/main/media/final

#AIChallenge24h #AI4LIFE #Rikkei

---

## Ảnh đính kèm

| Ảnh chụp bản Windows | Chú thích |
|---|---|
| [Giao diện ViCare](media/final/01-interface.png) | Người dùng nhập câu hỏi sức khỏe tiếng Việt và chọn Dense hoặc Hybrid. |
| [Câu trả lời có nguồn](media/final/02-answer-with-source.png) | Câu trả lời và đoạn được trích cùng nói về yếu tố ảnh hưởng tới kết quả đo lưu huyết não. |
| [Ngoài lĩnh vực](media/final/03-out-of-domain.png) | Câu hỏi về thủ đô nước Úc được trả về nhánh ngoài phạm vi sức khỏe. |
| [Mã nguồn truy xuất](media/final/04-retrieval-source.png) | Đường chạy dùng chung cho tra cứu và đánh giá. |

## Hai bước chủ dự án thực hiện

1. Sao chép phần bài đăng từ tiêu đề đến hết ba hashtag và đính kèm bốn ảnh vào [nhóm Mê IT Nghiện Code](https://www.facebook.com/groups/meitnghiencode).
2. Lấy liên kết bài vừa đăng, điền và gửi [form của BTC](https://m4beo6fqhrl.sg.larksuite.com/share/base/form/shrlgB534MeoVcQaAjNI8oP2cBe). Với bài chờ duyệt chưa lấy được liên kết, dùng liên kết trang cá nhân theo thông báo BTC đã cung cấp.

Hạn BTC thông báo là **12:00 ngày 23/11/2026, giờ Việt Nam (UTC+7)**.
