# 🚀 Project: Sentiment Analysis with Logistic Regression
## Ứng dụng Demo (Streamlit) cho Đề tài Cuối kỳ

Đây là một ứng dụng web được xây dựng bằng Streamlit để trình diễn khả năng của mô hình **Logistic Regression** (đã được huấn luyện trong Jupyter Notebook) trong việc phân loại cảm xúc (Tích cực/Tiêu cực) cho các đánh giá sản phẩm bằng Tiếng Anh.

Ứng dụng này tập trung 100% vào đề tài, cho thấy mô hình có thể được triển khai thành một công cụ thực tế với ba chế độ nhập liệu linh hoạt.

---

## 🧠 "Bộ Não" Cốt lõi của Ứng dụng

Toàn bộ ứng dụng được vận hành bởi "bộ não" gồm 2 file đã được huấn luyện:

1.  **`tfidf_vectorizer_english.pkl` (Người Phiên Dịch)**
    * **Nhiệm vụ:** Đây là bộ từ vựng `TfidfVectorizer` (với 5000 từ). Nó dịch các câu văn Tiếng Anh (chữ) thành các vector 5000 chiều (số) mà mô hình có thể hiểu được.

2.  **`sentiment_model_english.pkl` (Người Ra Quyết Định)**
    * **Nhiệm vụ:** Đây là mô hình `LogisticRegression` đã được huấn luyện. Nó nhận vector số từ "Người Phiên Dịch" và sử dụng các "trọng số" (weights) đã học để tính toán và ra phán quyết cuối cùng: Tích cực (1) hay Tiêu cực (0).

---

## ✨ Các Tính năng chính

Ứng dụng cung cấp 3 cách khác nhau để kiểm tra sức mạnh của cùng một mô hình:

### 1. Phân tích Trực tiếp (Câu đơn)
* **Mục đích:** Demo khả năng xử lý thời gian thực.
* **Cách hoạt động:** Người dùng nhập một câu đánh giá Tiếng Anh. Ứng dụng sẽ ngay lập tức làm sạch, vector hóa, và dự đoán cảm xúc kèm theo "Độ tự tin" (tính bằng hàm `predict_proba`).

### 2. Phân tích Dữ liệu Dán (Nhiều câu)
* **Mục đích:** Demo khả năng xử lý dữ liệu copy "live" từ web.
* **Cách hoạt động:** Người dùng dán một khối văn bản (nhiều bình luận, mỗi bình luận 1 dòng). Ứng dụng tự động tách từng dòng, phân tích chúng, và trả về một báo cáo tổng quan (biểu đồ tròn) và bảng kết quả chi tiết.

### 3. Phân tích Hàng loạt (File)
* **Mục đích:** Demo khả năng ứng dụng trong thực tế (xử lý dữ liệu lớn).
* **Cách hoạt động:** Người dùng tải lên một file `.csv` hoặc `.xlsx`. Ứng dụng sẽ phân tích toàn bộ các dòng (dựa trên cột được chọn), trả về báo cáo tổng quan và cho phép **tải về file kết quả** (CSV) đã được thêm cột "Sentiment_Label".

---

## 📁 Cấu trúc Thư mục

Để ứng dụng hoạt động, thư mục project (`D:\TEST_AI`) phải chứa các file sau:

D:\TEST_AI

│ ├── sentiment_model_english.pkl <-- (Bộ não - Người Ra Quyết Định) ├── tfidf_vectorizer_english.pkl <-- (Bộ não - Người Phiên Dịch) │ ├── app.py <-- (Code giao diện Streamlit) │ └── .venv/ <-- (Môi trường ảo)


---

## 🛠️ Cài đặt & Chạy Ứng dụng

### Bước 1: Tạo Môi trường ảo (Nếu chưa có)
Mở terminal trong thư mục `D:\TEST_AI` và gõ:
```bash
py -m venv .venv
Bước 2: Kích hoạt Môi trường
Bash

.\.venv\Scripts\activate
Bước 3: Cài đặt các Thư viện
(Bạn phải ở trong môi trường .venv khi chạy lệnh này)

Bash

pip install streamlit pandas plotly openpyxl nltk scikit-learn
Bước 4: Chạy Ứng dụng
Sau khi cài đặt xong, gõ lệnh sau để khởi chạy:

Bash

streamlit run app.py