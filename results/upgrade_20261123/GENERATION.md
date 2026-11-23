# Đối chiếu generation trước/sau — Linux CPU, 23/11/2026

**Bản cuối không chứng minh được tăng độ chính xác hoặc khả năng trả lời trên `dev[:10]`.** Cùng ngân sách 180 giây cho mỗi yêu cầu Ollama, baseline trả lời 3/10 câu; bản cuối trả lời 1/10. Cả hai vẫn được Judge chấm 10/10. Đối chiếu văn bản cho thấy bản cuối loại bỏ một câu trả lời lệch ngữ cảnh (`d003`), đồng thời làm mất một câu trả lời có thông tin hỗ trợ một phần (`d002`). Câu trả lời còn lại (`d000`) giữ nguyên giới hạn về độ đầy đủ.

Đây là **đối chiếu mức hỗ trợ của nguồn có AI hỗ trợ**, dựa trên câu hỏi, đáp án tham khảo, câu trả lời thực tế và các đoạn đã truy xuất. Đây **không phải chấm điểm độc lập của con người, thẩm định lâm sàng hoặc xác minh kiến thức y khoa ngoài bộ dữ liệu**. Báo cáo chỉ công bố ID và nhận xét ngắn, không chép nội dung ca bệnh hoặc toàn bộ câu hỏi/đáp án.

## 1. Cặp đo chính và khả năng đối chiếu

| Thuộc tính | Baseline | Bản cuối |
| --- | --- | --- |
| Artifact | [generation_before180.json](generation_before180.json) | [generation_final.json](generation_final.json) |
| Tên lượt chạy | `before180` | `final180` |
| Hoàn tất UTC | 2026-11-23 02:44:10 | 2026-11-23 02:51:50 |
| Nền tảng | Linux x86_64, Python 3.11.14, CPU | Cùng môi trường |
| Tập câu hỏi | Đúng `dev[:10]`, 10 câu | Cùng 10 câu và thứ tự |
| Timeout mỗi yêu cầu Ollama | 180 giây | 180 giây |
| Generator và Judge | `qwen3:4b-instruct` | Cùng model/digest |
| Retrieval trong ứng dụng | Hybrid `exact`, top 5 | Cùng backend và ngân sách |

Đã đối chiếu: model digest, dataset revision, corpus hash, hash thứ tự Dev100, embedding revision, prompt/options của Judge, giới hạn gold 1.500 ký tự, temperature, giới hạn đầu ra, ngưỡng dense và các ngân sách retrieval đều giống nhau giữa hai artifact. Mỗi câu hỏi thực tế cũng giống nhau. Chi tiết đầy đủ nằm trong JSON và [PROTOCOL.md](PROTOCOL.md).

Digest model:

```text
0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0
```

Hash `src/generation.py` của hai lượt:

```text
before180: d005e9712078fcef3031e0b80cc8f1a666535e87b431c04075ff2bf2a75690d0
final180:  2514269ab52f180a22a5d67484bf9f2d5a2e01c0958fb7d6563607718c9d4422
```

Đây là so sánh toàn bộ ứng dụng trước/sau, không phải một phép tách riêng tác động của prompt: code chọn đoạn đại diện của retrieval cũng thay đổi. Tuy nhiên, các nguồn liên quan đến `d000`, `d002`, `d003` và nguồn được duyệt của `d005` giữ nguyên nội dung giữa hai lượt. Sự thay đổi ở các ca này không thể được giải thích bằng việc vừa tìm ra một tài liệu chuẩn mới. Một số đoạn đại diện không được duyệt ở `d001` và `d005` có thay đổi; ID đoạn được lưu trong các dòng JSON.

## 2. Judge phải được đọc cùng trạng thái phản hồi

| Chỉ số, mẫu số 10 | Baseline 180 giây | Bản cuối 180 giây |
| --- | ---: | ---: |
| Legacy Judge | 10/10 | 10/10 |
| Trạng thái `answered` | 3/10 | 1/10 |
| `answered` và Judge = 1 | 3/10 | 1/10 |
| Trạng thái `refused` | 6/10 | 8/10 |
| `refused` nhưng Judge = 1 | 6 | 8 |
| Chuyển nhánh cấp cứu | 1 | 1 |
| Lỗi kỹ thuật của ứng dụng | 0 | 0 |
| Judge sai định dạng | 0 | 0 |
| Lỗi gọi Judge | 0 | 0 |
| Thời gian `ask` trung bình | 40,57 giây | 42,04 giây |
| Thời gian `ask` p95 | 62,87 giây | 62,15 giây |

`supported_answer_yield` trong evaluator mới là số dòng có `status == answered` và Judge = 1, chia cho **toàn bộ 10 câu**. Tên trường này không có nghĩa rằng câu trả lời đã được xác minh đúng y khoa hoặc đủ ý. Ví dụ, baseline có một câu lệch ngữ cảnh nhưng vẫn được Judge chấm 1. Baseline chưa có các trường chẩn đoán mới; các số tương ứng trong bảng được tính trực tiếp từ `rows` của artifact gốc, không sửa artifact để giả thành lượt đo mới.

Legacy Judge giữ nguyên cách tính để đối chiếu lịch sử. Nó có thể cho điểm tích cực với lời từ chối và nhánh cấp cứu; không được viết kết quả này thành “chatbot chính xác 100%”. Thời gian `ask` là thời gian của các nhánh xử lý khác nhau, có thể gồm nhiều yêu cầu model. Không diễn giải timeout 180 giây là thời gian chờ cho toàn bộ câu hỏi, hoặc suy ra cải thiện tốc độ độc lập với việc trả lời/từ chối.

## 3. Đối chiếu từng ID

“Hỗ trợ một phần” nghĩa là phần được trích có trong nguồn nhưng chưa bao phủ các nhánh/điều kiện quan trọng của yêu cầu. “Từ chối có cơ sở” chỉ nói về giới hạn bằng chứng cho câu trả lời được yêu cầu; không chứng nhận hệ thống đã xử lý lâm sàng đầy đủ. Không có câu trả lời nào trong cặp đo được đối chiếu ở đây là **đã đầy đủ mọi phần liên quan**.

| ID | Baseline → bản cuối | Nhận xét về mức hỗ trợ và giới hạn |
| --- | --- | --- |
| `d000` | Trả lời → trả lời | **Hỗ trợ một phần, không đổi.** Câu trả lời cuối giống hệt baseline và là trích dẫn nguyên văn đúng nguồn. Nó giữ điều kiện của nhánh được trích nhưng bỏ nhánh đối lập và phần đánh giá/sàng lọc có trong cùng nguồn. Không coi đây là câu trả lời cá thể đầy đủ hoặc một cải thiện mới. |
| `d001` | Từ chối → từ chối | **Từ chối có cơ sở cho kết luận cá thể được yêu cầu.** Nguồn được duyệt không xác lập đầy đủ bối cảnh và điều kiện tham chiếu áp dụng cho người hỏi. Có thông tin tổng quát liên quan trong các tài liệu, nhưng không đủ để tự kết luận tình trạng của người hỏi. |
| `d002` | Trả lời → từ chối | **Mất thông tin hỗ trợ một phần.** Baseline trích đúng một nhánh có điều kiện nhưng bỏ nhánh khác. Bản cuối không đưa ra thông tin này dù cùng nguồn được duyệt còn nguyên; toàn bộ nguồn gồm ba câu có thể biểu diễn các nhánh tham khảo trong giới hạn chọn đoạn. Không che việc giảm khả năng cung cấp thông tin bằng cách chỉ gọi phản hồi mới là “an toàn hơn”. |
| `d003` | Trả lời → từ chối | **Loại bỏ một câu trả lời lệch ngữ cảnh.** Trích dẫn baseline đúng nguyên văn nhưng dựa vào một tham chiếu ngược thuộc biểu hiện khác trong nguồn, không hỗ trợ trọn tổ hợp được hỏi. Các nguồn được duyệt không xác lập câu trả lời đầy đủ cho tổ hợp đó. Bản cuối không còn phát ra câu trả lời lệch phạm vi này. |
| `d004` | Từ chối → từ chối | **Còn mất phần thông tin có thể cung cấp.** Nguồn hỗ trợ đặc tính dùng một lần; điều đó **không chứng minh cơ chế tự hủy**. Không được đánh đồng hai khái niệm hoặc gọi đây là từ chối sai một đáp án đầy đủ. Thông tin hữu ích cùng nằm với ký hiệu thể tích/định dạng tách câu; ràng buộc trích nguyên văn và chặn liều lượng vẫn giới hạn cách trả lời. |
| `d005` | Từ chối → từ chối | **Từ chối có cơ sở đối với hướng điều trị cá thể.** Nguồn được duyệt mô tả người thân trong một ca khác; không được chuyển quan hệ gia đình, biểu hiện hoặc bệnh sử đó sang câu hỏi hiện tại. Bản cuối không lặp lại lỗi gán ngữ cảnh đã xuất hiện ở candidate v1. |
| `d006` | Từ chối → từ chối | **Từ chối có cơ sở cho đánh giá cá thể; còn khả năng cung cấp thông tin có điều kiện.** Nguồn được duyệt dựa trên tuổi/diễn tiến cụ thể không được xác lập trong câu hỏi. Không thể chuyển nguyên bối cảnh hoặc nguyên nhân của ca đó sang người hỏi. |
| `d007` | Từ chối → từ chối | **Từ chối có cơ sở do phạm vi/vị trí không được khớp đủ.** Nguồn được duyệt mô tả một vị trí và tổ hợp biểu hiện khác; không dùng sự gần nhau về từ vựng để xác định một bệnh cho câu hỏi hiện tại. |
| `d008` | Từ chối → từ chối | **Không đủ cơ sở cho một chỉ định chung; còn thông tin tham khảo có điều kiện.** Các đoạn được duyệt khác nhau về nguyên nhân, mức độ và điều kiện áp dụng; khả năng thực hiện không tự thành khuyến nghị thực hiện. Từ chối tránh đưa ra một lựa chọn chung, nhưng không cung cấp phần phân biệt có điều kiện sẵn có trong các nguồn. |
| `d009` | Cấp cứu → cấp cứu | **Nhánh định tuyến theo chính sách.** Hai lượt đều không truy xuất tài liệu và đưa cùng thông báo cấp cứu. Đây không phải câu trả lời kiến thức theo gold, cũng không chứng minh độ đúng của phân loại cấp cứu trên các tình huống khác. |

Ba câu trả lời baseline và câu trả lời duy nhất của bản cuối đều được đối chiếu là chuỗi nguyên văn có trong đoạn đã trích dẫn. Kết quả của `d003` cho thấy kiểm tra chuỗi và số nguồn hợp lệ vẫn chưa đủ để xác nhận đúng ngữ cảnh.

## 4. Candidate v1 không được chấp nhận, kết quả vẫn được giữ

[generation_after.json](generation_after.json) là candidate v1, tên lượt `after`, timeout **45 giây** mỗi yêu cầu. [generation_before.json](generation_before.json) là baseline thăm dò tương ứng ở 45 giây. Hai tệp được giữ nguyên; chúng không được thay bằng số của cặp đo cuối 180 giây.

| Candidate v1, 45 giây | Kết quả đã lưu |
| --- | ---: |
| Legacy Judge | 8/10 |
| Trả lời / từ chối / cấp cứu | 1 / 8 / 1 |
| Trả lời được Judge chấm 1 | 0/10 |
| Từ chối được Judge chấm 1 | 7 |
| Lỗi kỹ thuật / lỗi gọi Judge | 0 / 0 |

Câu trả lời của v1 ở `d005` đúng chuỗi nguồn nhưng đưa vào một quan hệ gia đình và bối cảnh không được thiết lập trong câu hỏi. Vì vậy không coi v1 là nâng cấp chất lượng. V1 cũng làm mất thông tin có nguồn ở `d000`/`d002`. Chẩn đoán riêng ghi nhận bộ đối chiếu có thể từ chối khi câu hỏi suy ngược khác cách hỏi gốc, dù đoạn được chọn hỗ trợ trực tiếp một câu hỏi tổng hợp đơn giản. Đó là cơ sở sửa cách kiểm tra phạm vi; không phải lý do nới điều kiện y khoa hoặc ép model trả lời.

Hash `src/generation.py` được artifact v1 ghi nhận:

```text
4a0a1db8aa8d98dc89415a31c3ee7cbed20dac266b261d5ec7cf6d8a93ccf679
```

## 5. Giới hạn còn lại và cách trình bày khi nộp

- **Độ đầy đủ vẫn yếu:** `d000` chỉ có một phần, `d002` mất thông tin so với baseline, `d004` và một số ca khác còn có thông tin có điều kiện chưa được trình bày. Cho phép ba câu liền nhau không bảo đảm model chọn một đoạn đủ ý.
- **Guard có đánh đổi:** chặn lời kể cá thể và tham chiếu ngược giúp tránh lấy ngữ cảnh của ca khác, nhưng có thể từ chối một câu hữu ích nếu câu đó phụ thuộc vào bối cảnh hoặc có cấu trúc lời khuyên ngắn. Regex không thay thế đối chiếu ý nghĩa.
- **Judge không phải nhãn đúng đáng tin cậy:** cùng model làm generator/judge, mẫu chỉ có 10 câu, và mọi lời từ chối của bản cuối vẫn được chấm 1. Các chỉ số yield mới chỉ tách trạng thái để người đọc không hiểu sai điểm Judge.
- **Tập đã được xem:** `dev[:10]` đã được dùng chẩn đoán và đối chiếu trong phát triển. Đây không phải kết quả trên holdout độc lập; chưa chứng minh chất lượng trên câu hỏi mới.
- **Nguồn/gold có giới hạn riêng:** câu trích có thể thuộc ca cá thể, thiếu bối cảnh hoặc chứa kiến thức cần được cập nhật. Đối chiếu với chúng không xác minh y khoa ngoài bộ dữ liệu. Ngưỡng cosine 0,86 vẫn là ngưỡng demo chưa được hiệu chỉnh lâm sàng.
- **Phạm vi thực thi:** các số trên là Linux CPU, không phải bằng chứng chạy trên hosted Colab hoặc trên GPU. Không trộn chúng với số Windows lịch sử để gán mức tăng cho thuật toán.

Cách mô tả phù hợp cho bản nộp: đã sửa tính nhất quán giữa đoạn được kiểm tra và đoạn hiển thị, thêm kiểm tra ngữ cảnh, và báo rõ khả năng trả lời/từ chối. **Không tuyên bố generation đã tăng độ chính xác hoặc đạt 100% đúng từ cặp đo này.**
