# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** 2A202603872
- **Họ và tên:** Mai Lê Bá Vương
- **SEED** (in ở ô đầu notebook): 3711

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính


| Mô hình                  | F1 test | F1 SUB 40 | p95 (ms/tin) | USD/1k tin |
| ------------------------ | ------- | --------- | ------------ | ---------- |
| 1. BoW + LogReg          | 0.879   | 0.919     | 0.124        | 0.000      |
| 2. BoW + MLP             | 0.933   | 0.923     | 5.192        | 0.000      |
| 3a. Embedding + kNN      | 0.877   | 0.927     | 45.841       | 0.000      |
| 3b. Embedding + LogReg   | 0.929   | 0.923     | 27.804       | 0.000      |
| 3c. Embedding + MLP head | 0.934   | 0.923     | 40.992       | 0.000      |
| 4a. LLM nhỏ zero-shot    | 0.930   | 0.930     | 1004.789     | 0.000      |
| 4b. LLM lớn zero-shot    | 0.952   | 0.952     | 400.890      | 0.000      |
| 4c. LLM nhỏ few-shot     | 0.952   | 0.952     | 1111.402     | 0.000      |


## 3. Dự đoán và thực tế

1\. Thí nghiệm 4.A — Model lớn vs nhỏ

- Dự đoán: Gemini lớn có F1 cao hơn khoảng 0.05–0.10.
- Thực tế: F1 model lớn = 0.952, model nhỏ = 0.930, chỉ chênh 0.022.
- Giải thích: Bài toán phân loại spam tương đối đơn giản nên model nhỏ đã hoạt động tốt, khiến lợi ích từ model lớn không đáng kể.

2\. Thí nghiệm 4.B — Few-shot

- Dự đoán: Thêm 4 ví dụ giúp F1 tăng 0.02–0.05.
- Thực tế: F1 tăng từ 0.930 lên 0.952 (+0.022), phù hợp dự đoán nhưng mức cải thiện nhỏ.
- Giải thích: Model đã hiểu nhiệm vụ zero-shot nên các ví dụ chỉ giúp cải thiện một số trường hợp.

3\. Stress test — 50 case khó

- Dự đoán: BoW đạt accuracy 50–65%, Embedding 65–80%, LLM 75–90%.
- Thực tế: BoW đạt 52–56%, Embedding 66–72%, LLM đạt 90–92%.
- Giải thích: LLM xử lý ngữ cảnh và các tin nhắn ngụy trang tốt hơn dự đoán, trong khi BoW phụ thuộc nhiều vào từ khóa.

## 4. Phân tích 50 case khó

Model tụt nhiều nhất: BoW + LogReg, F1 giảm từ 0.919 xuống 0.333 (−0.586). Nhóm khó nhất là spam giọng tự nhiên (10% đúng), thao túng/lừa đảo (12%) và né từ khóa (17%). Nguyên nhân là BoW dựa vào token rời rạc, không hiểu ngữ cảnh hoặc ý định lừa đảo.

Hai case cụ thể:

- Case #21: “Hi Mum, I dropped my phone ... Can you send £200 ...” — Spam giả danh người thân nhưng viết giống hội thoại bình thường, khiến BoW khó nhận diện vì thiếu từ khóa spam quen thuộc.
- Case #2: “FREE pizza in the office kitchen right now ...” — Tin ham nhưng chứa từ khóa “FREE” và lời giục giã. Model LLM phòng thủ phân loại sai thành spam do nhầm thông báo nội bộ với quảng cáo.

Kết luận: Test thông thường cho kết quả lạc quan vì cùng phân phối với dữ liệu train. Stress test giúp bộc lộ điểm yếu khi gặp spam mới hoặc tin nhắn cố tình đánh lừa.

## 5. Thử thách cá nhân

Đề số 4 (SEED = 3711): So sánh kNN với k = 1, 5, 25 và giải thích ảnh hưởng của k đến F1.

Cách làm: Sử dụng `knn_predict` trên embedding của tập test, thay đổi k và tính F1.

| Giá trị k | F1 test |
| --------- | ------- |
| 1         | 0.896   |
| 5         | 0.877   |
| 25        | 0.845   |

Giải thích: k = 1 đạt F1 cao nhất. Khi k tăng, F1 giảm do lấy thêm nhiều láng giềng không liên quan và dễ thiên về lớp ham chiếm đa số. Tuy nhiên, k quá nhỏ vẫn có nguy cơ nhạy với nhiễu và outlier.

## 6. Phương án cho sản phẩm thật

Tôi chọn BoW + MLP vì đạt F1 test = 0.933, F1 SUB = 0.923, p95 = 5.192 ms/tin và throughput khoảng 16.340 tin/giây, đáp ứng yêu cầu 10 triệu SMS/ngày và SLA dưới 50 ms.

So với LLM lớn (F1 SUB = 0.952, p95 = 400.890 ms), BoW + MLP có chất lượng thấp hơn một chút nhưng nhanh hơn nhiều và không phát sinh chi phí API cho mỗi tin. Tuy nhiên, cần cải thiện khả năng nhận diện spam ngụy trang vì F1 trên 50 case khó chỉ đạt 0.476.

## 7. AI log và điều đã học

| Phần | Bạn hỏi AI gì?                                                    | AI trả lời/viết gì?                                                                    | Bạn đã sửa / kiểm chứng gì?                                                                      |
| ---- | ----------------------------------------------------------------- | -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| 4    | Viết code gọi LLM API, tạo prompt và xử lý kết quả phân loại spam | Code `call_openai`, `build_classification_prompt`, `parse_label`, `llm_classify_batch` | Chạy unit test, kiểm tra bằng Gemini API và sửa lỗi rate limit, content filter, reasoning effort |

Điều đã học: Tôi hiểu rằng LLM có thể phân loại spam mà không cần huấn luyện lại, chỉ cần thiết kế prompt phù hợp. Tuy nhiên, khi sử dụng API thực tế còn phải xử lý giới hạn request, lỗi từ bộ lọc an toàn và chi phí token. Ngoài ra, few-shot có thể cải thiện F1 nhưng làm tăng chi phí do prompt dài hơn.