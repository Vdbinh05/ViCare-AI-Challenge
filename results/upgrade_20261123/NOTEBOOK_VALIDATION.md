# Kiểm chứng nội dung notebook — 23/11/2026

**15/15 ô code được thực thi tuần tự thành công**, và báo cáo phiên xác nhận mọi stage bắt buộc đạt. Lượt này chạy bằng **IPython InteractiveShell.run_cell trong cùng tiến trình, Linux CPU**, với môi trường/model/index đã có cache. **Chưa chạy Jupyter kernel thành công và chưa chạy trên Google Colab hosted.**

Thời gian thực: **10:02:35–10:17:21, ngày 23/11/2026, UTC+7**. Mã phiên: `20261123T030235Z-0e47e8`.

Lượt Google Colab hosted được ghi trước đó dùng source Git `b9a586a` của bản cũ. Bản nâng cấp trong báo cáo này dùng snapshot `21545d22…`, đã xác minh bằng output Linux CPU. Tình trạng hosted của snapshot này là **NOT_RUN**. [Lượt hosted của bản cũ](../colab/REPORT.md)

## File và tính toàn vẹn

- [Notebook chính](../../ViCare_Vuong_Dinh_Binh.ipynb): GPU mặc định, timeout 180, không điền sẵn output. SHA-256 `6c90f278373269b457996a532d1ab4e6fa8a401ab0bc494a4414c0c16dcdd7c8`.
- Snapshot source trong cả hai file: `21545d22b4ea5816d07775e663414472ceef0ae7cb2bce4ecde11f62195f60f5`. Tất cả 58 file nhúng được đối chiếu đúng byte với source đã khóa; source không đổi khi chạy.
- [Bản có output Linux CPU](../../ViCare_Linux_CPU_Executed.ipynb): SHA-256 `c8ac2c61847845ae9383abc3e76541e1dbd2c0625442a22f845aaa0d546923fc`. Các code cell, execution count và output giữ nguyên từ lượt chạy; chỉ thêm một ô giới thiệu và metadata mô tả môi trường thật. Bản này giữ `REQUIRE_GPU=False`, không đổi sang GPU trong khi giữ output CPU.
- Bản có output lưu hash canonical, hash raw và phương thức thực thi trong metadata; execution count từ 1 đến 15 và output hiện diện trên đủ 15 ô code. Sidecar gốc `notebook_validation.json` được báo cáo tác giả mô tả nhưng không nằm trong bộ tệp bàn giao. [Notebook có output](../../ViCare_Linux_CPU_Executed.ipynb)

## Kết quả từ chính source nhúng

Các ô đã tự kiểm tra môi trường, dựng/kiểm tra 11.227 đoạn, đo retrieval, khởi động đúng Qwen và sinh thật, đánh giá generation, chạy demo, hỏi đáp tương tác và tạo ZIP của phiên. Dev100 tái lập Hybrid exact **38/52/55/60 trên 100 câu tại @1/3/5/10**, Dense **36/46/50/54**. Confirmation100 exact tái lập **48/62/68/75**.

Generation tái lập **1 trả lời, 8 từ chối, 1 cảnh báo cấp cứu**, Legacy Judge 10/10, lỗi kỹ thuật 0. Notebook hiển thị riêng tám lời từ chối được judge chấm 1 và cảnh báo không coi điểm đó là độ chính xác. Bộ demo đạt **6/6**; ô tương tác mặc định thực sự trả lời từ nguồn về kết quả đo lưu huyết não. Các chỉ số và lời từ chối không được thay bằng output của lượt khác.

Lượt tương tác tạo tiến trình mới ghi thời gian `ask` **112.32 giây** trên CPU; toàn bộ ô mất 116,11 giây. Giữ nguyên thời gian thực này, không lấy số demo đã nạp sẵn hoặc độ trễ truy xuất riêng thay vào. Đoạn trích khớp hash câu nguồn đã duyệt, có trích dẫn [1] và 5 thẻ nguồn.

Pytest bên trong notebook ghi **`242 passed, 5 skipped in 15.28s`**: các cờ gọi model/index thật được tắt ở riêng ô regression; các ô sau dùng index/model thật. Bộ source ngoài notebook trước đó đã đạt **247/247, không bỏ qua**, với hai cờ thật bật; hai phép kiểm tra này khác phạm vi, không cộng số test với nhau. Xem [REPORT.md](REPORT.md) và [nhật ký stage pytest cuối](final_execution.json).

| Ô code | Trạng thái | Giây |
|---|---|---:|
| `configuration` | PASS | 0.01 |
| `restore-source` | PASS | 0.05 |
| `inspect-source` | PASS | 0.04 |
| `install-environment` | PASS | 0.74 |
| `regression-checks` | PASS | 17.02 |
| `build-index` | PASS | 16.01 |
| `historical-selection` | PASS | 0.01 |
| `evaluate-retrieval` | PASS | 81.15 |
| `evaluate-confirmation` | PASS | 50.99 |
| `prepare-ollama` | PASS | 13.28 |
| `evaluate-generation` | PASS | 453.36 |
| `run-demo-cases` | PASS | 137.42 |
| `interactive-question` | PASS | 116.11 |
| `export-results` | PASS | 0.03 |
| `archived-evidence` | PASS | 0.01 |

Ô `export-results` của notebook có output ghi đã tạo ZIP phiên và báo đủ các stage. ZIP 11 tệp không đi kèm bản bàn giao. Báo cáo gốc ghi SHA-256 `222525c8ca15c530921a4c6fc2d8c8788601440a2f653bd7c07420f1f89216d3`; không thể đối chiếu trực tiếp digest của ZIP từ các tệp hiện có. [Xem output của phiên](../../ViCare_Linux_CPU_Executed.ipynb)

## Giới hạn và lượt bị chặn

Lượt nbclient/Jupyter ban đầu chết trước ô đầu vì thao tác dò giao diện mạng và bind socket bị môi trường từ chối. Một smoke dùng transport IPC được Jupyter hỗ trợ cũng bị từ chối bind Unix socket, trước khi chạy câu lệnh thử. Không thay quyền, namespace, proxy hoặc cơ chế kiểm soát để vượt chặn. Sau đó kiểm chứng trực tiếp các ô bằng IPython trong cùng tiến trình; đây là cách thực thi khác và được ghi rõ trong metadata.

`capture_output` gom stdout/stderr và rich display theo từng ô, không bảo toàn thứ tự xen kẽ. Log subprocess thừa kế file descriptor có thể chỉ vào log runner. Đây là giới hạn ghi output; các bảng và kết quả được tạo từ artifact thực của phiên.

Lượt này không kiểm chứng quota GPU, lần tải sạch, giao diện Colab, khả năng chia sẻ công khai hoặc thời lượng trên dịch vụ Google. Output được rà soát bằng công cụ AI trước khi công khai: không có toàn bộ câu hỏi/gold hay đoạn ca bệnh của Dev10; demo dùng câu hỏi tổng hợp, đoạn trả lời công khai khớp hash câu nguồn đã duyệt. Việc này không thay thế thẩm định y khoa. **Trạng thái hosted Colab vẫn là NOT_RUN.**

Hai sidecar kiểm chứng này không được nhúng ngược vào notebook, tránh thay hash của chính file vừa kiểm tra. Không rebuild notebook chỉ để thêm báo cáo này.
