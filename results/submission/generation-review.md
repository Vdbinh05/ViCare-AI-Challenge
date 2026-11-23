# Đánh giá sinh câu trả lời theo mục 8

**Kết quả cuối đạt Judge@10 là 9/10. Ứng dụng trả lời 1 câu, từ chối 8 câu và chuyển hướng cấp cứu 1 câu.** Giám khảo cho 1 điểm ở câu có trả lời, 7/8 lần từ chối và lần chuyển hướng cấp cứu. Trong hai trường hợp từ chối dù nguồn đầu có thông tin trực tiếp, giám khảo cho một điểm 0 và một điểm 1. Tỷ lệ 90% vì thế không thể được hiểu là 90% câu hỏi nhận được câu trả lời đúng.

## Cách đo

Chạy từ thư mục gốc dự án bằng PowerShell.

```powershell
.\.venv\Scripts\python.exe -X utf8 -u -m benchmarks.evaluate_generation
```

Lệnh hoàn tất với mã thoát 0 ở cả hai lượt được báo cáo. Trước lượt đầu, một lần khởi động dừng ở bước nạp E5 do Windows thiếu bộ nhớ cấp phát (OS error 1455), trước khi xử lý câu đầu tiên. Sau khi phiên ứng dụng dùng để kiểm thử đã đóng, cả mười câu được chạy trong một tiến trình cho mỗi phiên bản.

Mẫu gồm đúng `dev[:10]` theo thứ tự `d000`–`d009` của Dev100 lịch sử, lấy từ validation với seed 2026. Script kiểm tra dấu vân tay của toàn bộ danh sách câu hỏi và kho tài liệu. Đây là tập phát triển từng được phân tích. Chỉ câu hỏi và tên phương pháp Hybrid được truyền vào `Assistant.ask`. Đáp án tham chiếu chỉ được dùng sau khi ứng dụng đã trả lời.

E5, Chroma và Ollama đều chạy cục bộ. Hybrid dùng cấu hình đã khóa với độ sâu Dense 200, nhóm ứng viên 80, RRF 40, trọng số hai nhánh bằng nhau và hiển thị tối đa năm nguồn. Ngưỡng cosine 0,86 của bản demo chưa được hiệu chuẩn lâm sàng. Cùng model `qwen3:4b-instruct` đảm nhận hai vai trò sinh và chấm. Script giữ nguyên prompt mục 8, cắt gold ở 1.500 ký tự, truyền nguyên văn phản hồi ứng dụng đến giám khảo, đặt nhiệt độ 0 và giới hạn 8 token. Chỉ một ký tự `0` hoặc `1` sau khi bỏ khoảng trắng đầu cuối mới hợp lệ. Mọi lượt lỗi hoặc sai định dạng đều tính 0 trên mẫu số cố định 10.

| Chỉ số | Kết quả |
|---|---:|
| Điểm 1 / tổng số câu | 9/10 |
| Điểm 0 / không hợp lệ / lỗi gọi judge | 1 / 0 / 0 |
| Trả lời / từ chối / chuyển cấp cứu | 1 / 8 / 1 |
| Lỗi truy xuất hoặc sinh gây gián đoạn | 0 |
| Tỷ lệ từ chối theo trạng thái ứng dụng | 8/10 |
| Đoạn trả lời trùng nguyên văn nguồn trích | 1/1 câu có trả lời |
| Có tài liệu gold trong năm kết quả thực trả | 4/9 lượt có truy xuất |

Câu cấp cứu đi qua nhánh bảo vệ trước truy xuất. Chuỗi từ chối nguyên văn của notebook xuất hiện 0/10 lần vì giao diện dùng thông báo tiếng Việt khác. Bảng đếm tám lượt theo trạng thái `refused`. JSON cũng lưu riêng kết quả đếm theo chuỗi cũ và model digest. [Kết quả số và ID tài liệu](generation.json)

## Đối sánh hai lượt cục bộ

| Cấu hình | Đúng theo judge | Trả lời | Từ chối | Chuyển cấp cứu | Lượt chấm không hợp lệ |
|---|---:|---:|---:|---:|---:|
| Trước khi sửa kiểm tra phạm vi | 9/10 | 3 | 6 | 1 | 0 |
| Sau khi sửa kiểm tra phạm vi | 9/10 | 1 | 8 | 1 | 0 |

Lượt đầu phát hiện câu trích nói về vị trí giải phẫu khác ở d007 và chỉ nêu tính khả thi trong khi người hỏi muốn biết có nên can thiệp ở d008. Chốt kiểm tra mới yêu cầu giữ hướng/vị trí được nêu tường minh và phân biệt yêu cầu khuyến nghị với câu hỏi về khả năng thực hiện. Cặp kiểm thử tổng hợp ngoài Dev10 dùng cùng một nguồn chỉ mô tả khả năng ghi nhật ký giấc ngủ. Với Ollama thật, hệ thống trả lời câu hỏi về khả năng và từ chối câu hỏi về khuyến nghị. Các câu kiểm tra riêng về phép đo và khả năng tiêm có điều kiện tiếp tục trả lời với nguồn tương ứng `train-4308` và `train-5104`.

Sau khi khóa sửa đổi, chạy lại đúng Dev10 một lần. Model, prompt judge, tham số truy xuất, ID câu hỏi và thứ tự tài liệu thu được giống nhau. Chỉ d007 và d008 chuyển từ trả lời sang từ chối. Cả hai lần chấm, mỗi câu ấy đều nhận 1 điểm. Tập này đã tham gia phát hiện lỗi nên kết quả sau sửa chỉ là đối chiếu trên tập đã xem. [Kết quả trước sửa](generation-before-scope.json)

## Rà soát từng câu

Đây là rà soát ngữ nghĩa của tác nhân thực hiện, chưa có thẩm định viên y tế. Bảng đối chiếu phạm vi câu hỏi, ý chính của đáp án tham chiếu, nội dung truy xuất và phần trích mà người dùng thực thấy. Tên đối tượng, số đo và bệnh sử cá nhân được lược bỏ. Hạng gold là hạng tài liệu đúng đầu tiên trong lượt truy xuất của ứng dụng.

| ID | Hạng gold | Hành vi | Judge | Nhận định |
|---|---:|---|---:|---|
| d000 | 1 | Từ chối | 0 | **Từ chối sai rõ.** Nguồn đầu nêu điều kiện để cân nhắc tiêm và yêu cầu sàng lọc cho đúng tình trạng được hỏi. Có thể trả lời có điều kiện mà không kết luận cho cá nhân. |
| d001 | — | Từ chối | 1 | Các nguồn nói về xét nghiệm khác. Thiếu khoảng tham chiếu và ngữ cảnh để diễn giải chỉ số được hỏi. Từ chối hợp lý, nhưng không giải thích các giới hạn có trong gold. |
| d002 | 1 | Trả lời | 1 | Nguồn được trích đúng. Câu trả lời giữ điều kiện về một cơ chế di truyền và nêu việc sàng lọc. Gold phân biệt thêm một cơ chế có hệ quả tư vấn khác. Người đọc chỉ nhận được một phần câu trả lời. |
| d003 | — | Từ chối | 1 | Các đoạn trích đề cập riêng từng triệu chứng, thiếu cơ sở đánh giá mức độ nguy hiểm của tổ hợp được hỏi. Gold cũng yêu cầu đánh giá rộng hơn. Từ chối hợp lý. |
| d004 | 1 | Từ chối | 1 | **Từ chối sai rõ.** Nguồn đầu giải thích trực tiếp đặc điểm dùng một lần của dụng cụ được hỏi. Câu hỏi về thiết bị có thể trả lời từ nguồn. Judge vẫn chấm 1 cho câu từ chối chung. |
| d005 | 2 | Từ chối | 1 | Từ chối đưa ra cách điều trị cá nhân là thận trọng. Gold trong top 5 có thông tin về giới hạn chẩn đoán và nhu cầu khám. Ứng dụng có thể chuyển tải nội dung này nhưng đã bỏ qua. |
| d006 | — | Từ chối | 1 | Gold vắng trong top 5. Một số đoạn vẫn có thông tin chung về hành vi thường gặp và khi nào cần đánh giá thêm. Chưa đủ để kết luận cho cá nhân. Phản hồi mất cơ hội giải thích có điều kiện. |
| d007 | — | Từ chối | 1 | Bản cũ trích một khả năng ở vị trí giải phẫu khác. Bản cuối từ chối vì câu trong nguồn thiếu giới hạn vị trí đã nêu. Gold mô tả khả năng khác. |
| d008 | — | Từ chối | 1 | Đoạn được chọn ở bản cũ chỉ xác nhận thủ thuật có thể thực hiện. Gold phân biệt nguyên nhân, khả năng tự hồi phục và nhu cầu khám. Bản cuối từ chối, tránh biến khả năng thành một khuyến nghị. |
| d009 | Không truy xuất | Cấp cứu | 1 | Bộ bảo vệ nhận diện triệu chứng cảnh báo và chuyển hướng ngay. Gold trình bày nhiều nguyên nhân và cách đánh giá. Điểm 1 không xác nhận hai phản hồi tương đương hoặc việc phân loại cấp cứu phù hợp ở mọi bối cảnh. |

## Diễn giải

Hai lần từ chối sai rõ đều có tài liệu gold ở hạng 1. Cải thiện truy xuất riêng lẻ sẽ không giải quyết được các trường hợp ấy. Câu d002 trích đúng nguồn nhưng chỉ bao quát một trong các điều kiện của đáp án. Chốt mới ưu tiên giữ giới hạn tường minh của nguồn. Câu hỏi hoặc nguồn diễn đạt bằng từ đồng nghĩa, hoặc cần kết hợp nhiều câu, vẫn có thể bị từ chối sai. Không thay ngưỡng cosine hoặc chọn lại cấu hình truy xuất từ Dev10.

Notebook gốc lưu kết quả lịch sử 7/10, ba lượt từ chối và không có lượt chấm không hợp lệ. Notebook cấu hình `qwen2.5:3b`, dùng bộ sinh/truy xuất khác và đếm từ chối bằng một chuỗi cố định. Lần đo hiện tại dùng `qwen3:4b-instruct` cho cả sinh và chấm. Chưa có kết quả chấm lại hai tập dự đoán bằng cùng một giám khảo với cùng định nghĩa từ chối. Hai con số không xác lập mức cải thiện. Cả hai phép đo đều nhỏ và chịu thiên lệch tự chấm.
