# Chất lượng câu trả lời trên dev[:10]

**Cả Dense, Hybrid HNSW và Hybrid cosine chính xác đều được Judge chấm 9/10. Mỗi cấu hình từ chối 9 câu, chuyển cấp cứu 1 câu và trả lời có nguồn 0 câu.** Judge cho 1 điểm ở 8 lượt từ chối và lượt cấp cứu. Phép đổi HNSW sang cosine chính xác không làm thay đổi đoạn truy xuất hoặc phản hồi trên 10 câu này.

| Lượt Windows | Judge / 10 | Trả lời / từ chối / cấp cứu | Judge sai định dạng / lỗi gọi judge / lỗi ứng dụng | Có gold trong 9 lượt truy xuất | Độ trễ ask trung vị / p95 | Độ trễ judge trung vị / p95 |
|---|---:|---:|---:|---:|---:|---:|
| [Dense HNSW](dense.json) | 9 | 0 / 9 / 1 | 0 / 0 / 0 | 4 | 2,31 / 3,57 giây | 135 / 179 ms |
| [Hybrid HNSW đã khóa](locked.json) | 9 | 0 / 9 / 1 | 0 / 0 / 0 | 4 | 2,26 / 3,09 giây | 66 / 104 ms |
| [Hybrid cosine chính xác được chọn](selected.json) | 9 | 0 / 9 / 1 | 0 / 0 / 0 | 4 | 2,21 / 3,07 giây | 64 / 137 ms |

Mỗi cấu hình có một lượt hoàn chỉnh, theo thứ tự trong bảng. Một lượt Dense trước đó bị gián đoạn trên máy Windows trước khi hoàn tất. Kết quả trong bảng lấy từ một lượt mới đủ 10 câu. Thời gian của 10 câu có độ bất định lớn. Chữ ký dataset, hash prompt, digest mô hình và các thông số sinh/chấm giống nhau trên cả ba lượt. Khác biệt chính giữa hai Hybrid nằm ở tìm kiếm trên cùng vector E5. Cả hai lấy 200 đoạn dense, giữ 80, kết hợp với 80 tài liệu BM25 và đưa tối đa 5 nguồn vào ứng dụng. Dense lấy 100 đoạn và gộp theo tài liệu.

## Phép đo

Mười câu d000–d009 là mười phần tử đầu của Dev100 theo thứ tự seed 2026. Dev100 đã tham gia phân tích lỗi trước đây. Benchmark kiểm tra chữ ký kho, chỉ mục và toàn bộ Dev100. Với mỗi câu, benchmark truyền văn bản câu hỏi và tên phương pháp cho chính `Assistant.ask`. Chỉ evaluator đọc nhãn chuẩn. Bộ kiểm tra đủ bằng chứng, ngưỡng cosine 0,86, lựa chọn câu nguồn, kiểm tra phạm vi, từ chối và chuyển cấp cứu đều chạy như demo.

Dense và các cấu hình Hybrid dùng cùng `qwen3:4b-instruct`. Judge nhận câu hỏi, tối đa 1.500 ký tự của đáp án tham chiếu và toàn bộ câu trả lời người dùng thấy. Prompt mục 8 có SHA-256 `ead398d1be6e99839262ae652bc4a727f9c46942203170a21fe8597e19754e70`, nhiệt độ 0 và tối đa 8 token. Một điểm hợp lệ phải là đúng một ký tự 0 hoặc 1 sau khi bỏ khoảng trắng hai đầu. Mẫu số luôn bằng 10. Điểm không hợp lệ, lỗi chấm hoặc lỗi ứng dụng tính 0. JSON ghi riêng điểm trả về từ judge và điểm được tính. Trạng thái `refused` tính vào tỷ lệ từ chối. Hướng dẫn cấp cứu và lỗi kỹ thuật có cột riêng.

Thời gian `ask` tính từ lúc giao câu hỏi cho ứng dụng cho đến lúc có phản hồi, sau khởi tạo chỉ mục và kiểm tra Ollama. Thời gian judge được đo riêng. Hai vai trò sinh và chấm dùng cùng trọng số nên có thể tự thiên lệch. Đối chiếu bởi agent dựa trên câu hỏi, tham chiếu, đoạn thực trả và các trích dẫn. Độ đầy đủ là mức bao quát các ý chính trong tham chiếu. Một ý có thể được nguồn hỗ trợ nhưng câu trả lời vẫn bỏ sót ý quan trọng. Chưa có đánh giá của chuyên gia y tế.

## Đối chiếu bởi agent

Bảng ghi hạng tài liệu chuẩn trong **các nguồn ứng dụng thực lấy**, theo thứ tự Dense / Hybrid khóa / Hybrid chọn. Dấu “–” nghĩa là ngoài top 5. Hai Hybrid trả cùng thứ tự đoạn và cùng quyết định ở cả 10 câu. Tệp CSV ghi nhận định cho từng nguồn ở từng cấu hình, gồm 135 lượt đoạn truy xuất, 65 cặp câu hỏi–đoạn khác nhau. Mỗi mục phân biệt văn bản được truy xuất, văn bản qua cổng bằng chứng và văn bản có được người dùng nhìn thấy trong một câu trả lời có trích dẫn hay không. [Đối chiếu từng nguồn](source-review.csv)

| ID | Hạng gold | Judge | Đối chiếu câu hỏi, nguồn và phản hồi bản chọn |
|---|---|---:|---|
| d000 | 1 / 1 / 1 | 0 | **Từ chối sai.** Nguồn `validation-122#0` chứa phân nhánh theo mức độ và bước sàng lọc trong đúng bệnh cảnh. Cổng chấp nhận nguồn này. Phản hồi cuối bỏ toàn bộ nội dung có thể giải thích có điều kiện. |
| d001 | – / – / – | 1 | Thiếu khoảng tham chiếu của lần xét nghiệm để kết luận cho cá nhân. Nguồn `train-521#0` vẫn giải thích sự khác biệt giữa thiết bị và cơ sở. Cổng chỉ chấp nhận `train-6457#0`, chứa khoảng chung và một trường hợp khác. Lời từ chối bỏ mất lý do cụ thể. |
| d002 | 2 / 1 / 1 | 1 | **Từ chối sai.** Hai cơ chế di truyền và hệ quả cho thai kỳ sau đều ở trong cùng `validation-515#0`. Cổng chấp nhận đoạn này trong cả ba lượt. Phản hồi cuối bỏ cả hai ý. Thêm đoạn lân cận sẽ không cung cấp thêm hai ý vốn đã có. |
| d003 | – / – / – | 1 | Năm nguồn Hybrid đề cập chóng mặt khi đổi tư thế hoặc những triệu chứng khác. Không nguồn nào đủ để đánh giá mức độ nguy hiểm của toàn bộ tổ hợp được hỏi. Cổng Hybrid chấp nhận cả năm. Dense coi các nguồn là mâu thuẫn, trong khi sự khác biệt về bệnh cảnh chưa đủ chứng minh mâu thuẫn trực tiếp. Từ chối kết luận là hợp lý. |
| d004 | 1 / 1 / 1 | 1 | **Từ chối sai.** `validation-525#0` mô tả trực tiếp đặc điểm dùng một lần của dụng cụ được hỏi. Cổng chấp nhận đúng nguồn, ứng dụng bỏ câu trả lời và judge vẫn cho 1. Các nguồn về lịch tiêm hoặc phản ứng sau tiêm không giải đáp đặc điểm dụng cụ. |
| d005 | – / 2 / 2 | 1 | Hybrid lấy được gold `validation-663#0` nhưng cổng chỉ giữ `train-61#0`. Cả hai giải thích cần đánh giá nguyên nhân trước khi lựa chọn điều trị. Gold có thêm tiền sử không nằm trong câu hỏi ngắn. Tránh đưa phác đồ cá nhân là phù hợp. Lời từ chối chung bỏ mất lý do cần thăm khám. |
| d006 | – / – / – | 1 | `test-560#0` được cổng giữ lại nhưng nói về khóc đêm kéo dài ở nhóm tuổi cụ thể. Nguồn `train-79#0` nêu một số hoàn cảnh thường gặp. Có thể trình bày giới hạn của thông tin chung. Không đủ để khẳng định tình trạng của trẻ trong câu hỏi. |
| d007 | 3 / – / – | 1 | Cổng cả ba lượt chỉ giữ `test-283#0`, nói về vùng đáy lưỡi với biểu hiện khác. Dense có gold `validation-976#0` ở hạng 3 nhưng cổng bỏ qua. Ở Hybrid, `train-865#0` cùng vùng dưới lưỡi nhưng giả thiết trong đó phụ thuộc tình trạng không đau. Từ chối ngăn việc gán sai vị trí hoặc điều kiện. |
| d008 | – / – / – | 1 | Các đoạn không thống nhất về độ rộng của lời khuyên. `test-1972#0` khuyến nghị mổ rộng, `validation-693#0` nêu khả năng thực hiện, `train-5095#0` và `train-4001#0` đặt điều kiện theo tổn thương. Cổng giữ cả bốn. Từ chối tránh khuyến nghị vô điều kiện, nhưng không truyền tải sự phụ thuộc vào bệnh cảnh hay nhu cầu đánh giá. |
| d009 | Bỏ qua truy xuất | 1 | Bộ bảo vệ chuyển hướng cấp cứu sớm. Không tài liệu nào được truy xuất hoặc trích dẫn. Phản hồi đưa ra hành động khẩn cấp, trong khi tham chiếu liệt kê các nguyên nhân và cách đánh giá. |

**Mức hỗ trợ và độ đầy đủ.** Bản chọn không có câu trả lời y khoa được trích nguồn trong Dev10. Do đó chưa có mẫu để tính tỷ lệ chính xác của phát biểu được trích. Không câu nào bao quát đầy đủ nội dung tham chiếu. Ở d000, d002, d004, cổng bằng chứng đã duyệt chính đoạn gold ở hạng 1, sau đó tầng sinh hoặc kiểm tra đầu ra vẫn từ chối. Dấu vết đo được không xác định quy tắc con nào gây từ chối. Cả 9 câu có truy xuất đều nhận `SUFFICIENT` ở tầng bằng chứng của hai Hybrid. Trường hợp d003 và d007 minh họa cổng còn duyệt những đoạn thiếu bệnh cảnh. Hạ ngưỡng để trả lời thêm sẽ không giải quyết được hai loại lỗi này.

Lượt lịch sử dùng Hybrid khóa từng trả lời một phần d002, chỉ đề cập một cơ chế dù đoạn nguồn đã có cả hai. Lượt đo lại dùng cùng digest mô hình, hash prompt judge và cùng thứ tự ID nguồn nhưng từ chối câu đó. Chưa xác định được nguyên nhân của sự khác biệt giữa hai lượt. Dùng kết quả ba lượt hiện tại để so baseline với bản chọn. Chưa ghi nhận câu nào có **chất lượng phản hồi** tăng trên Dev10. [Phép đo lịch sử](../../submission/generation-review.md)

## Kiểm tra demo bằng câu tổng hợp

Sáu ca riêng trên Windows đều đạt kiểm tra hành vi. Câu hỏi chung về các yếu tố ảnh hưởng phép đo lưu huyết não nhận được một trả lời có trích dẫn từ `train-4308`. Câu hỏi về yếu tố liên quan tăng huyết áp bị từ chối, tránh lấy nhầm mô tả phép đo để trả lời. Câu hỏi địa lý được trả về `OUT_OF_DOMAIN`. Câu hỏi về thuốc giả định nhận `INSUFFICIENT_EVIDENCE`. Khi trỏ riêng trình sinh vào một cổng loopback không lắng nghe, ứng dụng báo `GENERATION_FAILURE` và vẫn hiển thị 5 nguồn đã truy xuất. Câu tổng hợp về bất tỉnh ở cùng điều kiện nhận hướng dẫn cấp cứu với 0 nguồn. Sáu ca này dùng mẫu số riêng với Dev10. [Kết quả và nguồn công khai của các câu tổng hợp](demo.json)
