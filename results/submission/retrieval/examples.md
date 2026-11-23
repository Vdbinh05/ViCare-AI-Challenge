# Trường hợp quan sát được

Mỗi dòng là một câu dev thật, định danh theo CSV. Thành công nghĩa là tìm thấy ít nhất một tài liệu chuẩn trong 10 tài liệu đầu. Dùng mã bộ dữ liệu để đối chiếu mà không sao chép nội dung câu hỏi hoặc tài liệu.

## Hybrid tìm thấy tài liệu chuẩn

- `d001` (`validation-328`), tài liệu chuẩn `validation-328`. Dense hit@10 = 0, Hybrid hit@10 = 1. Top 3 Dense `train-5669, test-1708, train-5007`. Top 3 Hybrid `train-5007, validation-956, train-6457`.
- `d027` (`validation-452`), tài liệu chuẩn `validation-452`. Dense hit@10 = 0, Hybrid hit@10 = 1. Top 3 Dense `validation-672, test-701, test-349`. Top 3 Hybrid `validation-672, test-701, validation-581`.
- `d028` (`validation-246`), tài liệu chuẩn `validation-246`. Dense hit@10 = 0, Hybrid hit@10 = 1. Top 3 Dense `train-2155, test-1487, train-2630`. Top 3 Hybrid `validation-246, test-346, test-1018`.

## Hybrid bỏ sót tài liệu chuẩn

- `d008` (`validation-106`), tài liệu chuẩn `validation-106`. Dense hit@10 = 1, Hybrid hit@10 = 0. Top 3 Dense `test-1972, train-4001, validation-693`. Top 3 Hybrid `test-1972, validation-693, train-5095`.
- `d072` (`validation-728`), tài liệu chuẩn `validation-728`. Dense hit@10 = 1, Hybrid hit@10 = 0. Top 3 Dense `validation-256, train-2102, train-4190`. Top 3 Hybrid `train-4190, train-2102, train-6654`.
- `d003` (`validation-975`), tài liệu chuẩn `validation-975`. Dense hit@10 = 0, Hybrid hit@10 = 0. Top 3 Dense `train-3673, train-1497, train-2472`. Top 3 Hybrid `validation-255, train-5874, train-3673`.
