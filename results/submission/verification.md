# Kiểm chứng trên máy cục bộ

Ngày 23/11/2026 (GMT+7). Kiểm thử dùng Python 3.11, E5 đã lưu cục bộ, Chroma hiện có và mô hình Ollama `qwen3:4b-instruct`. Các lệnh dưới đây chạy từ thư mục dự án. Trên máy kiểm thử, thư mục tạm của tiến trình được đặt trên ổ đĩa dự án.

## Môi trường và lần chạy đầu

Lượt đầu đo retrieval dừng khi Windows không nạp được PyTorch (WinError 1114); bộ nhớ cam kết còn khoảng 1,8 GiB. Khi các tiến trình kiểm thử đã giải phóng bộ nhớ, cùng lệnh hoàn tất với mã 0. Lượt generation đầu cũng dừng trước câu thứ nhất vì thiếu bộ nhớ ảo (WinError 1455). Các phép đo hoàn tất được chạy tuần tự. [Thông số retrieval](retrieval/evaluation.json)

Trước bản sửa cuối ở bước đối chiếu ý hỏi, chạy toàn bộ pytest với `VICARE_QA_REAL_INDEX=1` và `VICARE_QA_REAL_OLLAMA=1` cho 159 đạt, 0 lỗi, 0 bỏ qua trong 30,39 giây. Các ca tích hợp dùng dữ liệu, E5, Chroma và Ollama thật. Các ca đơn vị kiểm lỗi dịch vụ, trích dẫn, input, HTML/URL và chỉ dẫn nằm trong nguồn.

## Mất kết nối và phục hồi qua trình duyệt

Chrome headless 158 gửi các yêu cầu tổng hợp qua giao diện Gradio thật. Một proxy loopback riêng chuyển tiếp tới Ollama cục bộ. Kiểm thử chỉ ngắt proxy, giữ nguyên daemon, mô hình và chỉ mục. Trình duyệt đối chiếu PID đang lắng nghe trước và sau các lượt. Toàn bộ phép kiểm này thoát mã 0.

| Tình huống | Kết quả quan sát |
|---|---|
| Ngắt proxy sau khi ứng dụng sẵn sàng | `POST /api/chat` bị từ chối kết nối (WinError 10061) sau 2,031 giây. UI báo dịch vụ chưa sẵn sàng và vẫn hiển thị đủ 5 thẻ nguồn. |
| Hai yêu cầu cấp cứu, có dấu và không dấu, trong lúc gián đoạn | Cả hai lần hiển thị cảnh báo gọi 115 và 0 thẻ nguồn. Bộ đếm tại ranh giới hàm cho tổng hai lần cho kết quả 0 truy xuất, 0 gọi chat, 0 kiểm tra danh sách mô hình. |
| Mở lại đúng cổng proxy | Cùng PID và URL ứng dụng trả lời lại trong 2,02 giây, hiển thị 5 nguồn và đánh dấu trích dẫn [1]. |

Các lượt trên xác nhận hành vi kỹ thuật quan sát được trên máy kiểm thử. Chưa có xác nhận lâm sàng hoặc chấp nhận của người dùng cuối trong phép đo này.

## Kiểm tra bản nộp sau sửa bước đối chiếu

Từ một thư mục trống, xuất mã và test đã stage bằng `git checkout-index`. Dùng lại interpreter Python 3.11 của dự án, cache E5 và bộ ViHealthQA đã tải, chỉ mục Chroma và daemon Ollama sẵn có. Chạy từ thư mục xuất mã, kiểm tra Python đang import đúng bản sao, bật cả hai cờ tích hợp và tắt truy cập mạng Hugging Face. Toàn bộ suite trả **166 đạt, 0 lỗi, 0 bỏ qua**, 26,25 giây, mã thoát 0. Các ca thêm kiểm phân biệt khuyến nghị với khả năng thực hiện, cùng giới hạn vị trí của bằng chứng.

Một PowerShell mới mở bên ngoài thư mục repo chạy lệnh tuyệt đối ở README mà không đặt biến môi trường phục vụ kiểm thử. ViCare dùng lại Ollama, hoàn tất lượt chat sẵn sàng, chọn cổng khác hai cổng đã có tiến trình. Đã khớp PID trong terminal với listener và nhận HTTP 200 ở `/config`.

Chrome 158 gửi đủ **10/10 ca tổng hợp** trên phiên này, mã thoát 0: cảm lạnh, câu hỏi huyết áp ở Dense và Hybrid, thuốc giả định, thủ đô, liều dùng cá nhân, hai cách gõ câu cấp cứu, input trống và phép đo lưu huyết não. Hai câu huyết áp cùng bị từ chối. Câu về phép đo được trả lời trong 2,05 giây, đánh dấu đúng nguồn [1] `train-4308`. Đã kiểm gửi, xóa hội thoại rồi chụp lại ảnh từ phiên này.

Với ZYX-9000, ứng dụng hiển thị đúng thông báo:

> Tôi chưa tìm thấy đủ thông tin trong các tài liệu tham khảo để trả lời câu hỏi này một cách đáng tin cậy. Bạn có thể hỏi cụ thể hơn hoặc tham khảo nhân viên y tế có chuyên môn.

Ảnh ngoài phạm vi dùng một câu hỏi riêng về thủ đô nước Úc. [Giao diện](../../media/01-interface.png) · [Kết quả có nguồn](../../media/02-answer-with-source.png) · [Ngoài phạm vi](../../media/03-out-of-domain.png) · [Mã truy xuất](../../media/04-retrieval-source.png)
