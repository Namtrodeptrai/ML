# Dự án phát hiện SMS Spam bằng mô hình lai CNN và SVM (Hybrid CNN-SVM)

Dự án này triển khai một giải pháp phân loại tin nhắn rác (SMS Spam Detection) kết hợp giữa hai thuật toán Học máy: **CNN (Convolutional Neural Network)** đóng vai trò trích xuất đặc trưng tự động nâng cao từ chuỗi văn bản, và **SVM (Support Vector Machine)** làm nhiệm vụ phân loại tối ưu trên không gian đặc trưng thu được. 

Dự án cũng tích hợp sẵn giao diện Web (Frontend) trực quan và thẩm mỹ cao dựa trên kiến trúc Flask để người dùng dễ dàng sử dụng và thử nghiệm.

---

## 📂 Cấu trúc thư mục dự án

```text
spam-detection/
│
├── data/
│   ├── raw/                  # Tệp dữ liệu SMS gốc chưa xử lý
│   ├── processed/            # Tập dữ liệu đã làm sạch
│   └── split/                # Tập dữ liệu đã chia tách (Train/Val/Test)
│
├── notebooks/
│   ├── eda.ipynb             # Jupyter Notebook khám phá dữ liệu trực quan
│   └── experiments.ipynb     # Jupyter Notebook thử nghiệm nhanh từng bước
│
├── models/
│   ├── cnn_weights.weights.h5 # Trọng số CNN tốt nhất lưu lại
│   ├── svm_model.pkl         # Mô hình SVM đã huấn luyện
│   └── tokenizer.pkl         # Bộ tokenizer đã mã hóa từ
│
├── src/
│   ├── utils.py              # Các hàm tiện ích hỗ trợ đọc cấu hình và log
│   ├── download_data.py      # Tự động tải và giải nén dữ liệu từ UCI
│   ├── preprocessing.py      # Làm sạch dữ liệu và tách tập dữ liệu
│   ├── feature_extraction.py # Lớp bọc bộ Tokenizer và padding chuỗi số
│   ├── cnn_model.py          # Định nghĩa mạng CNN & Trích xuất đặc trưng
│   ├── svm_model.py          # Lớp bọc thuật toán phân loại SVM
│   ├── train.py              # Pipeline chính liên kết huấn luyện từ đầu đến cuối
│   ├── evaluate.py           # Tính toán các chỉ số và vẽ ma trận nhầm lẫn
│   ├── predict.py            # Chạy dự đoán tin nhắn mới từ giao diện CLI
│   └── app.py                # Máy chủ API Web Server bằng Flask
│
├── templates/
│   └── index.html            # Giao diện Web người dùng (HTML/CSS/JS)
│
├── tests/
│   ├── test_preprocessing.py # Kiểm thử bộ tiền xử lý
│   ├── test_model.py         # Kiểm thử cấu trúc mạng CNN
│   └── test_predict.py       # Kiểm thử chức năng suy luận (predict)
│
├── .gitignore                # Cấu hình bỏ qua các file tạm trước khi đẩy GitHub
├── config.yaml               # Cấu hình siêu tham số và đường dẫn
├── requirements.txt          # Các thư viện Python cần thiết
└── README.md                 # Tài liệu hướng dẫn sử dụng (tệp này)
```

---

## 🛠️ Hướng dẫn cài đặt và chạy nhanh (Khuyên dùng)

Dự án đã được tích hợp sẵn kịch bản chạy tự động (setup & run) và đi kèm **mô hình đã huấn luyện sẵn** (nằm trong thư mục `models/`). Bạn không cần tự cài đặt môi trường hay huấn luyện lại từ đầu.

### Trên Windows:
Bạn chỉ cần nhấp đúp chuột vào tệp **`run.bat`** (hoặc chạy lệnh sau trong CMD/PowerShell tại thư mục dự án):
```bash
run.bat
```
Hệ thống sẽ tự động khởi tạo môi trường ảo Python `.venv`, cài đặt các thư viện cần thiết, khởi chạy Flask Server và tự động mở trình duyệt web hiển thị giao diện tại địa chỉ: **[http://127.0.0.1:5000](http://127.0.0.1:5000)**.

### Trên macOS / Linux:
Mở Terminal tại thư mục dự án và chạy lệnh sau (lần đầu tiên có thể cần cấp quyền thực thi `chmod +x run.sh`):
```bash
chmod +x run.sh
./run.sh
```
Hệ thống sẽ tự động cài đặt môi trường và mở trang web tương tự như trên Windows.

---

## 🚀 Quy trình thực thi thủ công hoặc Huấn luyện lại mô hình

Nếu bạn muốn chỉnh sửa cấu hình, huấn luyện lại mô hình hoặc đánh giá chi tiết:

### Bước 1: Chuẩn bị môi trường thủ công
Nếu không dùng script tự động, bạn có thể tự cài đặt:
```bash
python -m venv .venv
# Kích hoạt trên Windows:
.venv\Scripts\activate
# Kích hoạt trên macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### Bước 2: Tải dữ liệu thô từ UCI
Tự động tải và giải nén tệp dữ liệu gốc SMS Spam từ kho UCI:
```bash
python src/download_data.py
```

### Bước 3: Huấn luyện lại mô hình từ đầu
Huấn luyện mạng CNN để trích xuất đặc trưng và kết hợp với SVM để phân loại:
```bash
python src/train.py
```

### Bước 4: Đánh giá mô hình trên tập kiểm thử (Test Set)
Tính toán các chỉ số (Accuracy, Precision, Recall, F1-score) và lưu biểu đồ ma trận nhầm lẫn tại `models/confusion_matrix.png`:
```bash
python src/evaluate.py
```

### Bước 5: Chạy dự đoán tin nhắn mới từ dòng lệnh (CLI)
```bash
python src/predict.py --text "Congratulations! You've won a $1000 gift card. Call 1800-999-9999 to claim your reward now."
```

---

## 🧠 Nguyên lý hoạt động của Mô hình lai (CNN + SVM)

1. **Chuẩn hóa văn bản**: Tin nhắn được làm sạch (chuyển chữ thường, xóa ký tự đặc biệt, lọc stop words) để giảm nhiễu.
2. **Biểu diễn văn bản**: Sử dụng `Tokenizer` để chuyển đổi các từ thành các ID số nguyên và thực hiện padding để đảm bảo độ dài chuỗi cố định.
3. **CNN (Convolutional Neural Network)**: Mạng CNN chứa tầng `Embedding` kết hợp với tầng tích chập `Conv1D` và `GlobalMaxPooling1D` để trích xuất các đặc trưng ngữ cảnh/cụm từ cục bộ quan trọng nhất.
4. **Feature Extraction**: CNN được huấn luyện trước trên tác vụ phân loại nhị phân. Khi đạt hiệu năng tốt nhất, tầng phân loại cuối cùng bị loại bỏ. Chúng ta đưa chuỗi số đi qua CNN để thu về vector đặc trưng 64 chiều (từ tầng Dense ẩn).
5. **SVM (Support Vector Machine)**: Nhận các vector 64 chiều từ CNN làm đầu vào và tìm siêu phẳng (hyperplane) tối ưu để tách biệt hai lớp Ham (tin nhắn thường) và Spam (tin nhắn rác) với biên lớn nhất.
