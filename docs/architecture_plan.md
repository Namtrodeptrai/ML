# Kế hoạch & Kiến trúc hệ thống phân loại SMS Spam (CNN + SVM)

Tài liệu này trình bày sơ đồ phân tích và luồng xử lý chi tiết của dự án nhận diện tin nhắn rác sử dụng mô hình lai giữa mạng nơ-ron tích chập (CNN) và máy vectơ hỗ trợ (SVM).

---

## I. Tổng quan kiến trúc mô hình lai (Hybrid CNN-SVM)

Sự kết hợp giữa **CNN** và **SVM** tận dụng thế mạnh lớn nhất của cả hai phương pháp:
1. **CNN (Convolutional Neural Network)**: Rất mạnh trong việc tự động tìm hiểu, trích xuất cấu trúc cú pháp và ý nghĩa ngữ cảnh từ chuỗi văn bản (Feature Extraction) mà không cần con người định nghĩa các bộ lọc từ khóa thủ công.
2. **SVM (Support Vector Machine)**: Rất mạnh trong việc tìm ra ranh giới phân loại tối ưu (Maximum Margin) giữa hai lớp dữ liệu trong không gian nhiều chiều, đặc biệt hiệu quả trên các tập dữ liệu kích thước nhỏ đến vừa (Classification).

---

## II. Sơ đồ luồng xử lý tin nhắn

Dưới đây là sơ đồ mô tả hành trình của một tin nhắn từ khi được nhập vào hệ thống cho đến khi đưa ra nhãn kết quả cuối cùng:

```text
[ Tin nhắn thô nhập từ giao diện ]
                 │
                 ▼
 1. TIỀN XỬ LÝ (Preprocessing)
 ├─ Chuyển về chữ thường (lowercase)
 ├─ Xóa các ký tự đặc biệt, dấu câu, số
 └─ Loại bỏ từ dừng (Stop words - các từ vô nghĩa)
                 │
                 ▼
    [ Văn bản đã làm sạch ]
                 │
                 ▼
 2. MÃ HÓA & CĂN CHỈNH ĐỘ DÀI (Tokenizer & Padding)
 ├─ Tokenizer: Ánh xạ từ vựng sang mã số nguyên định danh
 └─ Padding: Chuẩn hóa chiều dài tin nhắn về đúng 80 ký tự số
                 │
                 ▼
   [ Chuỗi số nguyên kích thước 80 ]
                 │
                 ▼
 3. TRÍCH XUẤT ĐẶC TRƯNG BẰNG MẠNG CNN
 ├─ Embedding Layer: Ánh xạ số nguyên sang vector thực 100 chiều
 ├─ Conv1D Layer: Quét bộ lọc kích thước 5 để học ngữ cảnh cục bộ
 ├─ GlobalMaxPooling1D: Giữ lại đặc trưng nổi bật nhất của câu
 └─ Dense Layer (Tầng ẩn): Rút gọn thông tin về Vector 64 chiều
                 │
                 ▼
    [ Vector đặc trưng 64 chiều ]
                 │
                 ▼
 4. PHÂN LOẠI TỐI ƯU BẰNG SVM
 ├─ Nhận vector 64 chiều đại diện cho tin nhắn làm đầu vào
 ├─ Hàm Kernel (RBF): Chiếu vector lên không gian phân tách tối ưu
 └─ Siêu phẳng (Hyperplane): Chia tách biên lớn nhất thành 2 vùng độc lập
                 │
                 ▼
[ Kết quả đầu ra: HAM (Tin thường) hoặc SPAM (Tin rác) ]
```

---

## III. Chi tiết các bước xử lý và đường đi của tin nhắn

### 1. Bước tiền xử lý (Preprocessing)
*   **Ví dụ đầu vào:** *"URGENT! You have won a 1-week FREE membership! Call 1800-999 to claim."*
*   **Mục đích:** Giảm thiểu độ nhiễu của câu, tập trung vào những từ thực sự mang giá trị phân loại.
*   **Hành động:** Hệ thống chuyển toàn bộ về chữ thường, lọc bỏ các ký tự đặc biệt như `!`, `-`, `.`, và loại bỏ các từ dừng không quan trọng (như *a, an, the, is*).
*   **Kết quả:** `"urgent won week free membership call claim"`

### 2. Bước Tokenizer & Padding
*   **Mục đích:** Chuyển đổi dữ liệu chữ viết thành cấu trúc số học để máy tính tính toán.
*   **Hành động:** 
    *   **Tokenizer:** Tra cứu từng từ trong từ điển đã học lúc huấn luyện. Ví dụ: `urgent` $\rightarrow$ `15`, `free` $\rightarrow$ `3`, `membership` $\rightarrow$ `45`.
    *   **Padding:** Chuẩn hóa độ dài câu về đúng **80 từ** (`max_len`). Nếu câu ngắn hơn, thêm các số `0` vào đầu câu.
*   **Kết quả:** Một mảng số nguyên: `[0, 0, ..., 0, 15, 89, 3, 45, 220]`

### 3. Trích xuất đặc trưng bằng CNN (Feature Extraction)
*   Mảng 80 số nguyên đi qua mạng CNN để thực hiện các phép toán:
    *   **Embedding Layer:** Biến mỗi số nguyên thành một vector 100 chiều liên tục, giúp giữ lại quan hệ ngữ nghĩa (ví dụ: từ `free` và `gift` có khoảng cách gần nhau).
    *   **Conv1D Layer (Tích chập):** Sử dụng các bộ quét trượt qua các cụm từ liền kề nhằm phát hiện các tổ hợp từ mang tính chất Spam (như cụm `"free membership"` đứng cạnh nhau).
    *   **GlobalMaxPooling1D:** Trích lọc các đặc trưng mạnh nhất thu được từ bước tích chập.
    *   **Dense Layer:** Đưa qua tầng ẩn tuyến tính để giảm chiều dữ liệu về **64 chiều**. 
*   **Kết quả:** Một vector số thực 64 chiều tóm tắt hoàn hảo toàn bộ ý nghĩa ngữ cảnh của tin nhắn ban đầu.

### 4. Phân loại bằng bộ phân loại SVM
*   **Hành động:** Nhận vector đặc trưng 64 chiều từ CNN. Bộ phân loại SVM sử dụng hàm hạt nhân RBF (Radial Basis Function) để vẽ một ranh giới (siêu phẳng phân tách - separating hyperplane) trong không gian 64 chiều này sao cho khoảng cách từ ranh giới đó tới các điểm tin nhắn dữ liệu đã biết là xa nhất có thể.
*   **Quyết định:** 
    *   Nếu điểm dữ liệu của tin nhắn mới nằm ở phía bên này ranh giới: Hệ thống dán nhãn **HAM (Tin thường)**.
    *   Nếu điểm dữ liệu nằm ở phía đối diện: Hệ thống dán nhãn **SPAM (Tin rác)**.

---

## IV. Vai trò và tác dụng của từng thuật toán

*   **Mạng CNN (Đóng vai trò trích xuất đặc trưng)**: Giải quyết điểm yếu lớn nhất của các mô hình học máy truyền thống là phải thiết lập thủ công các từ khóa (TF-IDF). CNN tự động học và hiểu được trật tự từ cũng như ngữ cảnh linh hoạt của câu.
*   **Thuật toán SVM (Đóng vai trò phân loại)**: Giải quyết điểm yếu của mạng nơ-ron truyền thống (như tầng Softmax cuối câu dễ bị Overfitting hoặc tìm ranh giới cục bộ không tối ưu). SVM tìm kiếm một ranh giới phân biệt lý tưởng nhất giữa 2 nhóm, tối đa hóa độ chính xác và đảm bảo mô hình hoạt động ổn định trên cả dữ liệu mới chưa từng thấy.
