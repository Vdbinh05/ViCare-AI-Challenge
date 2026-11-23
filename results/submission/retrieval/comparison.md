# So sánh truy xuất Dense và Hybrid

Dev100 gồm 100 câu ViHealthQA có mã tài liệu chuẩn trong Chroma. Mỗi điểm phần trăm trong bảng ứng với một câu.

| Chỉ số | Dense | Hybrid | Hybrid − Dense |
|---|---:|---:|---:|
| Recall@1 | 37.0% | 38.0% | +1.0 đ.% |
| Recall@3 | 47.0% | 52.0% | +5.0 đ.% |
| Recall@5 | 51.0% | 55.0% | +4.0 đ.% |
| Recall@10 | 55.0% | 60.0% | +5.0 đ.% |
| Coverage | 100.0% | 100.0% | +0.0 đ.% |
| Trung bình (ms) | 24.45 | 130.01 | +105.56 ms |
| Trung vị (ms) | 23.89 | 89.48 | +65.59 ms |
| p95 (ms) | 37.22 | 276.11 | +238.89 ms |

Đo trên 100 câu dev theo thứ tự đã xác minh, không gọi LLM.
Dense lấy 100 đoạn khi đo top 10. Hybrid lấy 200 đoạn từ HNSW, giữ 80 đoạn theo cosine và lấy tối đa 80 tài liệu từ BM25. RRF hợp nhất theo mã tài liệu.
Cả hai phương pháp khởi động bằng ba câu ngoài dev. Thứ tự câu hỏi giống nhau và phương pháp chạy trước được luân phiên theo câu.

Ở @1/@3/@5/@10, Hybrid lần lượt thắng/thua Dense 7/6, 8/3, 6/2 và 7/2 câu. Coverage là tỷ lệ câu có ít nhất một tài liệu chuẩn trong chỉ mục dùng chung.

Giao diện mặc định yêu cầu 5 tài liệu. Với Dense, số ứng viên Chroma ở giao diện là 50. Hybrid giữ cùng ngân sách 200→80 đoạn và tối đa 80 tài liệu BM25. Recall@5 trong bảng được tính từ năm vị trí đầu của lần truy xuất top 10. Dev100 đã được dùng trong phân tích ở giai đoạn phát triển. Các đáp án tham chiếu nằm trong corpus. Phép đo phản ánh khả năng tìm lại tài liệu chuẩn. Đánh giá giá trị y khoa cần bằng chứng riêng.
